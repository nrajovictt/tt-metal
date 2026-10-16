#!/usr/bin/env python3
"""What-if: inside SDPA only QK^T and PV run on the FPU; everything else runs on the SFPU.

Moved to the SFPU (per 256-token K-chunk step, from sdpa_unit_split.py): the row-max reduce and its copy (72 tiles),
the subtract of sub+exp (64 tiles) and the correction's subtract / multiplies / add (56 tiles): ~192 element-wise tiles,
2,812 FPU cycles today (14.7 cycles per tile). Per-Q-chunk work (row sum, normalize) is <0.1% of time at 256-token
chunks and is ignored.

Cost of the moved work on the SFPU, two cases:
  same    - the same cycles it takes on the FPU today (upper bound on the benefit)
  sfpu    - 128 cycles per tile: 4 single-cycle SFPU instructions (load, load/op, op, store) per 32-lane slot
Stage wall time = measured wall - moved FPU cycles + their SFPU cost; waits on unpack/pack stay as measured.

Execution models, each with the measured exp and with the single-cycle exp of fast_exp_model.py:
  one thread  - Blackhole: stages back to back on one math thread
  two threads - Quasar: FPU thread runs QK^T and PV only; SFPU thread runs row max, sub+exp, correction.
                Deps: MAX_j<-QK_j,MAX_{j-1}; EX_j<-MAX_j; PV_j<-EX_j; CORR_j<-PV_j,CORR_{j-1}; QK_j<-PV_{j-2} (2 score buffers)
                Order: FPU QK_0, QK_1, PV_0, QK_2, PV_1 ...; SFPU MAX_0, EX_0, MAX_1, EX_1, CORR_0, MAX_2, EX_2, CORR_1 ...
Today's split (FPU also runs reduce / correction element-wise work) is quasar_model.simulate.

Per length (2k and up): SDPA time scales by step ratio; FPU and SFPU busy time in SDPA scale by their per-step busy
cycles; whole-layer utilization re-divided by the new layer time (with two threads FPU + SFPU can exceed 100%).

usage: python tools/tracy/prefill_study/sfpu_offload_model.py OUT.json SPLIT.json LAYER.json LAYER_UNITS.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quasar_model import simulate as simulate_today

SFPU_TILE = 128  # cycles per element-wise tile on the SFPU at one cycle per instruction
FAST_EXP_TILE = 128  # single-cycle exp, as in fast_exp_model.py


def simulate_matmul_only(cost, n=14):
    F = [("QK", 0), ("QK", 1)] + [t for j in range(n) for t in (("PV", j), ("QK", j + 2))]
    # CORR_j only rescales O / l and is not needed by EX_{j+1} (which needs MAX_{j+1} only), so it goes after it
    S = [("MAX", 0), ("EX", 0)] + [t for j in range(n) for t in (("MAX", j + 1), ("EX", j + 1), ("CO", j))]
    deps = {
        "QK": lambda j: [("PV", j - 2)],
        "MAX": lambda j: [("QK", j), ("MAX", j - 1)],
        "EX": lambda j: [("MAX", j)],
        "PV": lambda j: [("EX", j)],
        "CO": lambda j: [("PV", j), ("CO", j - 1)],
    }
    end, sched, free, idx, q = {}, [], {"F": 0, "S": 0}, {"F": 0, "S": 0}, {"F": F, "S": S}
    while idx["F"] < len(F) or idx["S"] < len(S):
        moved = False
        for t in ("F", "S"):
            if idx[t] >= len(q[t]):
                continue
            name, j = q[t][idx[t]]
            dd = [d for d in deps[name](j) if d[1] >= 0]
            if all(d in end for d in dd):
                st = max([free[t]] + [end[d] for d in dd])
                end[(name, j)] = free[t] = st + cost[name]
                sched.append((t, name, j, st, st + cost[name]))
                idx[t] += 1
                moved = True
        assert moved, "deadlock"
    ex = sorted(s[3] for s in sched if s[1] == "EX")
    return ex[-3] - ex[-4], sched


def main():
    out, split_json, layer_json, units_json = sys.argv[1:5]
    S = json.load(open(split_json))
    sp, e = S["split_256"], S["fpu_per_elt_tile"]
    tiles = {k: round(sp[k]["fpu"] / e) for k in ("MAX", "SUBEXP", "CORR")}  # 72, 64, 56
    mm_fpu = sp["QK"]["fpu"] + sp["PV"]["fpu"]
    small_sfpu = sp["MAX"]["sfpu"] + sp["CORR"]["sfpu"]
    res = {}
    for exp_case, exp_sfpu in (("measured_exp", sp["SUBEXP"]["sfpu"]), ("single_cycle_exp", 64 * FAST_EXP_TILE)):
        base = {k: v["cycles"] for k, v in sp.items()}
        base["SUBEXP"] = base["SUBEXP"] - sp["SUBEXP"]["sfpu"] + exp_sfpu
        # today's split
        t1 = sum(base.values())
        q_cost = {
            "QK": base["QK"],
            "RED": base["MAX"] - sp["MAX"]["sfpu"],
            "PV": base["PV"],
            "CF": base["CORR"] - sp["CORR"]["sfpu"],
            "MS": sp["MAX"]["sfpu"],
            "CE": sp["CORR"]["sfpu"],
            "EX": base["SUBEXP"],
        }
        t2, _ = simulate_today(q_cost)
        fpu_busy = mm_fpu + sum(sp[k]["fpu"] for k in tiles)
        sfpu_busy = small_sfpu + exp_sfpu
        res[f"{exp_case}/today"] = {"one_thread": t1, "two_threads": t2, "fpu_busy": fpu_busy, "sfpu_busy": sfpu_busy}
        for cost_case in ("same", "sfpu"):
            st = dict(base)
            moved = 0
            for k, n in tiles.items():
                c_new = sp[k]["fpu"] if cost_case == "same" else n * SFPU_TILE
                st[k] = st[k] - sp[k]["fpu"] + c_new
                moved += c_new
            t1 = sum(st.values())
            t2, sched = simulate_matmul_only(
                {"QK": st["QK"], "PV": st["PV"], "MAX": st["MAX"], "EX": st["SUBEXP"], "CO": st["CORR"]}
            )
            res[f"{exp_case}/matmul_only_{cost_case}"] = {
                "one_thread": t1,
                "two_threads": t2,
                "fpu_busy": mm_fpu,
                "sfpu_busy": sfpu_busy + moved,
                "stages": st,
                "schedule": [list(x) for x in sched],
            }
    today = res["measured_exp/today"]["one_thread"]
    print(
        f"{'scenario':<40}{'1 thread':>10}{'x':>7}{'2 threads':>11}{'x':>7}{'FPU busy':>10}{'SFPU busy':>11}   (cycles per 256-token chunk step)"
    )
    for k, v in res.items():
        print(
            f"{k:<40}{v['one_thread']:>10.0f}{today / v['one_thread']:>7.2f}{v['two_threads']:>11.0f}{today / v['two_threads']:>7.2f}"
            f"{v['fpu_busy']:>10.0f}{v['sfpu_busy']:>11.0f}   1T FPU {v['fpu_busy'] / v['one_thread'] * 100:5.1f}% SFPU {v['sfpu_busy'] / v['one_thread'] * 100:5.1f}%"
            f" | 2T FPU {v['fpu_busy'] / v['two_threads'] * 100:5.1f}% SFPU {v['sfpu_busy'] / v['two_threads'] * 100:5.1f}%"
        )

    units = {u["seq"]: u for u in json.load(open(units_json))}
    ref = res["measured_exp/today"]
    rows = []
    for lr in json.load(open(layer_json)):
        seq = lr["seq"]
        if seq < 2048:
            continue
        d = pd.read_csv(Path(layer_json).with_suffix("") / f"ops_{seq}.csv", low_memory=False)
        a, b = lr["layer_rows"]
        sd = d[(d.idx >= a) & (d.idx <= b) & (d.kind == "sdpa")].iloc[0]
        T, Ts, U = lr["layer"]["kernel_us"], sd.kern_ns / 1e3, units[seq]
        row = {"seq": seq, "layer_us": T, "fpu": U["fpu"], "sfpu": U["sfpu"]}
        for k, v in res.items():
            for model in ("one_thread", "two_threads"):
                r = v[model] / ref["one_thread"]
                Tn = T - Ts + Ts * r
                f_sd, s_sd = Ts * sd.FPU_act_mean, Ts * sd.SFPU_act_mean
                fpu_n = (U["fpu"] * T - f_sd + f_sd * v["fpu_busy"] / ref["fpu_busy"]) / Tn
                sfpu_n = (U["sfpu"] * T - s_sd + s_sd * v["sfpu_busy"] / ref["sfpu_busy"]) / Tn
                row[f"{k}/{model}"] = {"layer_us": round(Tn, 1), "fpu": round(fpu_n, 2), "sfpu": round(sfpu_n, 2)}
        rows.append(row)
        best = row["single_cycle_exp/matmul_only_same/two_threads"]
        print(
            f"  {seq:>6}: layer {T:10.1f} us; 2 threads + 1-cycle exp: today's split {row['single_cycle_exp/today/two_threads']['layer_us']:10.1f}, "
            f"matmul-only FPU {best['layer_us']:10.1f} us (FPU {best['fpu']:5.1f}%, SFPU {best['sfpu']:5.1f}%)"
        )
    Path(out).write_text(
        json.dumps({"tiles_moved": tiles, "sfpu_tile_cycles": SFPU_TILE, "scenarios": res, "rows": rows}, indent=1)
    )
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
