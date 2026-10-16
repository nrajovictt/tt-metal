#!/usr/bin/env python3
"""What-if: the softmax exp costs one cycle per SFPU instruction.

Assumption: exp is a single SFPU instruction and every instruction of the exp loop (load, 1/sqrt(d) multiply, exp,
store) takes one cycle per 32-lane slot: 4 x 32 = 128 cycles per 32x32 tile instead of the measured ~1,356
(sdpa_unit_split.py, scale multiply + accurate exp). Everything else stays as measured: the FPU subtract and the
unpack/pack waits of the sub+exp stage, the other SDPA stages, all other ops. Only the sub+exp exp changes (97% of
SDPA SFPU cycles); the correction's one-column exp, the row max and the reciprocal keep their measured cost.

Per prompt length (any chunk size), from the counters:
  exp share of SDPA SFPU = se*c / (se*c + small*s)      (se, small = per-step tile counts, c, s from the split)
  SDPA time saved        = T_sdpa * SFPU_busy(working cores) * exp share * (1 - new/c)
  layer: FPU busy time unchanged; SFPU busy time reduced by the same exp cycles (mean over active cores);
         utilizations re-divided by the new layer time.
Also: one-step timelines (Blackhole single math thread, and the Quasar two-thread model) with the cheaper exp.

usage: python tools/tracy/prefill_study/fast_exp_model.py OUT.json SPLIT.json LAYER.json LAYER_UNITS.json [CYCLES_PER_SLOT=4]
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quasar_model import simulate

HEADS = 32


def counts(seq):
    chunk = 256 if seq >= 2048 else 64
    sq = chunk // 32
    nq = -(-seq // chunk)
    iters, per_q = nq * (nq + 1) / 2, nq
    corr, kq = (iters - per_q) / iters, iters / per_q
    return chunk, sq * sq, sq * corr * 2 + sq / kq


def main():
    out, split_json, layer_json, units_json = sys.argv[1:5]
    per_slot = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    new_tile = per_slot * 32
    S = json.load(open(split_json))
    c, s, sp = S["sfpu_per_se_tile"], S["sfpu_per_small_op"], S["split_256"]

    # one 256-token chunk step, Blackhole single math thread
    old = {k: v["cycles"] for k, v in sp.items()}
    se = sp["SUBEXP"]
    new = dict(old, SUBEXP=se["cycles"] - se["sfpu"] + 64 * new_tile)
    bh_old, bh_new = sum(old.values()), sum(new.values())
    # Quasar two-thread model with the cheaper exp
    q_cost = {
        "QK": old["QK"],
        "RED": old["MAX"] - sp["MAX"]["sfpu"],
        "PV": old["PV"],
        "CF": old["CORR"] - sp["CORR"]["sfpu"],
        "MS": sp["MAX"]["sfpu"],
        "CE": sp["CORR"]["sfpu"],
        "EX": new["SUBEXP"],
    }
    q_old_cost = dict(q_cost, EX=old["SUBEXP"])
    q_old, _ = simulate(q_old_cost)
    q_new, q_sched = simulate(q_cost)
    print(
        f"step cycles (256-token chunks): Blackhole {bh_old} -> {bh_new} with {new_tile}-cycle exp tiles ({bh_old / bh_new:.2f}x); "
        f"Quasar model {q_old} -> {q_new} ({bh_old / q_new:.2f}x vs Blackhole today)"
    )
    print(
        f"  sub+exp stage {old['SUBEXP']} -> {new['SUBEXP']} cycles; SDPA FPU busy {sum(v['fpu'] for v in sp.values()) / bh_new * 100:.1f}% "
        f"of the new step, SFPU {(sum(v['sfpu'] for v in sp.values()) - se['sfpu'] + 64 * new_tile) / bh_new * 100:.1f}%"
    )

    units = {u["seq"]: u for u in json.load(open(units_json))}
    rows = []
    for lr in json.load(open(layer_json)):
        seq = lr["seq"]
        d = pd.read_csv(Path(layer_json).with_suffix("") / f"ops_{seq}.csv", low_memory=False)
        a, b = lr["layer_rows"]
        sd = d[(d.idx >= a) & (d.idx <= b) & (d.kind == "sdpa")].iloc[0]
        chunk, se_t, small = counts(seq)
        share = se_t * c / (se_t * c + small * s)
        cut = 1 - new_tile / c
        T, Ts = lr["layer"]["kernel_us"], sd.kern_ns / 1e3
        saved_t = Ts * sd.SFPU_work_med / 100 * share * cut  # time on the busiest (working) cores
        saved_busy = Ts * sd.SFPU_act_mean / 100 * share * cut  # in the metric's own units (mean over active cores)
        U = units[seq]
        Tn = T - saved_t
        fpu_n = U["fpu"] * T / Tn
        sfpu_n = (U["sfpu"] * T - saved_busy * 100) / Tn
        r = {
            "seq": seq,
            "chunk": chunk,
            "layer_us": T,
            "layer_fast_exp_us": round(Tn, 1),
            "layer_change_pct": round((Tn / T - 1) * 100, 1),
            "sdpa_us": round(Ts, 1),
            "sdpa_fast_exp_us": round(Ts - saved_t, 1),
            "fpu": U["fpu"],
            "sfpu": U["sfpu"],
            "fpu_fast_exp": round(fpu_n, 2),
            "sfpu_fast_exp": round(sfpu_n, 2),
            "either_fast_exp": round(fpu_n + sfpu_n, 2),
            "exp_share_of_sdpa_sfpu": round(share, 3),
        }
        if seq >= 2048:
            r["layer_quasar_fast_exp_us"] = round(T - Ts + Ts * q_new / bh_old, 1)
        rows.append(r)
        print(
            f"  {seq:>6}: layer {T:10.1f} -> {Tn:10.1f} us ({r['layer_change_pct']:+.1f}%)  SDPA {Ts:9.1f} -> {Ts - saved_t:9.1f}  "
            f"FPU {U['fpu']:5.2f} -> {fpu_n:5.2f}  SFPU {U['sfpu']:5.2f} -> {sfpu_n:5.2f}"
            + (f"  | Quasar+fast exp layer {r['layer_quasar_fast_exp_us']:.1f} us" if seq >= 2048 else "")
        )
    json.dump(
        {
            "cycles_per_tile_new": new_tile,
            "cycles_per_tile_measured": c,
            "stage_old": old,
            "stage_new": new,
            "bh_step_old": bh_old,
            "bh_step_new": bh_new,
            "quasar_step_old": q_old,
            "quasar_step_new": q_new,
            "quasar_cost": q_cost,
            "quasar_schedule": [list(x) for x in q_sched],
            "rows": rows,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
