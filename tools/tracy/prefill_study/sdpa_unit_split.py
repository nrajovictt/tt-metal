#!/usr/bin/env python3
"""Split SDPA's FPU / SFPU / idle cycles across its stages (one 256-token K-chunk step).

Inputs: stage means from sdpa_stages.py (needs one 256-token-chunk length and one 64-token-chunk length) and the SDPA
op's counters from the counter runs (ops csv columns "FPU/SFPU/MATH Util Avg (%)"). Counter semantics: FPU/SFPU = cycles
an instruction is valid on that unit, MATH = either; in this kernel FPU + SFPU == MATH (no overlap).

Method:
  * unit of each call is read from compute_common.hpp (matmul/reduce/eltwise binary -> FPU; exp, mul_unary, recip,
    binary_max -> SFPU; mask = tile copy + packer L1 accumulate)
  * matmul FPU cycles = tile products x 64 (HiFi4: 16 cycles per 32x32x32 tile product at LoFi, 4 fidelity passes)
  * SFPU cost per sub+exp tile (c) and per small SFPU tile op (s) solved from the two chunk sizes together:
        tiles_se * c + small_ops * s = SFPU cycles per step    (one equation per chunk size)
  * FPU cost per element-wise tile = (FPU cycles - matmul cycles) / element-wise tiles, at 256-token chunks
Per-stage split is only reported for 256-token chunks: at 64-token chunks the math thread's timestamps run ahead of the
hardware on short stages (PV measures 53 cycles per tile product, below the 64-cycle rate).

usage: python tools/tracy/prefill_study/sdpa_unit_split.py OUT.json STAGES.json \
          --counters 32768:generated/profiler/prefill_32k_1dev 1024:generated/profiler/prefill_1k_1dev
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sdpa_counters(run):
    rep = sorted(glob.glob(f"{run}/reports/*/"))[-1]
    ops = pd.read_csv(glob.glob(rep + "ops_perf_results_*.csv")[0], low_memory=False)
    s = ops[ops["OP CODE"].str.contains("SDPA") & ops["DEVICE KERNEL DURATION [ns]"].notna()].iloc[-1]
    return {k: float(s[f"{k} Util Avg (%)"]) / 100 for k in ("FPU", "SFPU", "MATH")}


def step_model(r, ctr):
    seq, chunk, freq = r["seq"], r["chunk"], r["freq_mhz"]
    st = {k[5:]: v["mean_us"] for k, v in r["riscs"]["math"]["stages"].items()}
    nq = -(-seq // chunk)
    iters, per_q = nq * (nq + 1) / 2, nq
    kq, corr = iters / per_q, (iters - per_q) / iters
    T = (
        st["QK"]
        + st["MAX"]
        + st["SUBEXP"]
        + st["PV"]
        + st["CORR"] * corr
        + (st.get("MASK", 0) + st.get("ROWSUM", 0) + st.get("NORM", 0)) / kq
    )
    cyc = T * freq
    sq, dh = chunk // 32, 4
    return {
        "seq": seq,
        "chunk": chunk,
        "freq": freq,
        "stage_us": st,
        "corr": corr,
        "step_us": T,
        "step_cycles": cyc,
        "fpu": ctr["FPU"] * cyc,
        "sfpu": ctr["SFPU"] * cyc,
        "idle": (1 - ctr["MATH"]) * cyc,
        "tile_products": sq * sq * dh + sq * dh * sq,
        "se_tiles": sq * sq,
        "small_sfpu": sq * corr + sq * corr + sq / kq,  # binary max, correction exp (1st column), recip
        "elt_tiles": sq * sq + sq * sq + sq * corr + (sq + sq + sq + sq * dh) * corr + (sq * dh + sq) / kq,
    }


def main():
    out, stages_json = sys.argv[1], sys.argv[2]
    ctr_runs = dict(a.split(":", 1) for a in sys.argv[sys.argv.index("--counters") + 1 :])
    R = {r["seq"]: r for r in json.load(open(stages_json))}
    M = [step_model(R[int(s)], sdpa_counters(run)) for s, run in ctr_runs.items()]
    big = next(m for m in M if m["chunk"] == 256)
    small = next(m for m in M if m["chunk"] == 64)
    A = np.array([[big["se_tiles"], big["small_sfpu"]], [small["se_tiles"], small["small_sfpu"]]])
    c, s = np.linalg.solve(A, [big["sfpu"], small["sfpu"]])
    e = (big["fpu"] - 64 * big["tile_products"]) / big["elt_tiles"]
    F, us = big["freq"], big["stage_us"]
    rows = {
        "QK": (us["QK"] * F, 64 * big["tile_products"] / 2, 0.0),
        "MAX": (us["MAX"] * F, (big["se_tiles"] + big["chunk"] // 32) * e, (big["chunk"] // 32) * s),
        "SUBEXP": (us["SUBEXP"] * F, big["se_tiles"] * e, big["se_tiles"] * c),
        "PV": (us["PV"] * F, 64 * big["tile_products"] / 2, 0.0),
        "CORR": (us["CORR"] * F, (big["chunk"] // 32 * 3 + big["chunk"] // 32 * 4) * e, (big["chunk"] // 32) * s),
    }
    print(
        f"SFPU per sub+exp tile {c:.0f} cyc, per small SFPU tile op {s:.0f} cyc; FPU per element-wise tile {e:.1f} cyc"
    )
    print(
        f"matmuls = {64 * big['tile_products'] / big['fpu'] * 100:.1f}% of FPU cycles (256-token chunks), "
        f"{64 * small['tile_products'] / small['fpu'] * 100:.1f}% (64-token chunks)"
    )
    split = {}
    for k, (cy, f, sf) in rows.items():
        split[k] = {"cycles": round(cy), "fpu": round(f), "sfpu": round(sf), "idle": round(cy - f - sf)}
        print(
            f"  {k:<7} {cy:8.0f} cyc | FPU {f:7.0f} ({f / cy * 100:5.1f}%) | SFPU {sf:7.0f} ({sf / cy * 100:5.1f}%) | idle {(cy - f - sf) / cy * 100:5.1f}%"
        )
    rf = sum(v["fpu"] for k, v in split.items() if k != "CORR") + split["CORR"]["fpu"] * big["corr"]
    rs = split["SUBEXP"]["sfpu"] + (split["MAX"]["sfpu"] + split["CORR"]["sfpu"]) * big["corr"]
    print(f"rebuilt per step: FPU {rf:.0f} vs counters {big['fpu']:.0f}; SFPU {rs:.0f} vs counters {big['sfpu']:.0f}")
    Path(out).write_text(
        json.dumps(
            {
                "sfpu_per_se_tile": c,
                "sfpu_per_small_op": s,
                "fpu_per_elt_tile": e,
                "step_cycles_256": big["step_cycles"],
                "split_256": split,
            },
            indent=1,
        )
    )
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
