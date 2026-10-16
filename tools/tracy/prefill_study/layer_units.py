#!/usr/bin/env python3
"""FPU and SFPU utilization of the whole layer (all ops, not only matmul / SDPA), per prompt length.

For each op in the layer scope (see layer_table.py): u_op = mean over the cores that ran its compute kernel of
counter / that core's counter window. Ops with no compute kernel (create/concat heads, KV-cache writes, pad, slice:
data movement only, verified to launch no TRISC kernel) count as 0. Layer value = sum(u_op * T_op) / sum(T_op),
T = device kernel time, the same weighting as the time-share chart. SDPA's own counters already include its
QK^T / PV matmuls, reduce and element-wise FPU work and its exp / max / reciprocal SFPU work.

Also reported: median-over-cores variant, core-time weighting (CORE COUNT x T), full-grid (110 cores, idle = 0),
and each contributor's points (matmul ops, SDPA, SiLU x multiply, other compute ops).

usage: python tools/tracy/prefill_study/layer_units.py LAYER.json OUT.json   (LAYER.json and its per-op CSVs from layer_table.py)
"""

import json
import sys
from pathlib import Path

import pandas as pd

COLS = [
    "FPU_act_mean",
    "FPU_act_med",
    "SFPU_act_mean",
    "SFPU_act_med",
    "MATH_act_mean",
    "FPU_full_mean",
    "SFPU_full_mean",
]


def group(r):
    if r.kind == "matmul":
        return "matmul"
    if r.kind == "sdpa":
        return "sdpa"
    if pd.isna(r.n_active):
        return "data_movement"
    if r.code == "BinaryNg" and r.SFPU_act_mean > 10:
        return "silu_mul"
    return "other"


def main():
    layer_json, out = Path(sys.argv[1]), sys.argv[2]
    res = []
    for lr in json.load(open(layer_json)):
        seq = lr["seq"]
        d = pd.read_csv(layer_json.with_suffix("") / f"ops_{seq}.csv", low_memory=False)
        a, b = lr["layer_rows"]
        L = d[(d.idx >= a) & (d.idx <= b)]
        L = L[(L.code == "Embeddings") | (L.idx >= L.idx[L.code == "LayerNorm"].min())].copy()
        L["group"] = [group(r) for r in L.itertuples()]
        L[COLS] = L[COLS].fillna(0.0)
        T = L.kern_ns.sum()
        w = L.kern_ns / T
        cw = L.kern_ns * L["CORE COUNT"]
        cw = cw / cw.sum()
        r = {
            "seq": seq,
            "layer_us": round(T / 1e3, 1),
            "fpu": round((L.FPU_act_mean * w).sum(), 2),
            "sfpu": round((L.SFPU_act_mean * w).sum(), 2),
            "either": round((L.MATH_act_mean * w).sum(), 2),
            "fpu_median_cores": round((L.FPU_act_med * w).sum(), 2),
            "sfpu_median_cores": round((L.SFPU_act_med * w).sum(), 2),
            "fpu_core_time": round((L.FPU_act_mean * cw).sum(), 2),
            "sfpu_core_time": round((L.SFPU_act_mean * cw).sum(), 2),
            "fpu_full_grid": round((L.FPU_full_mean * w).sum(), 2),
            "sfpu_full_grid": round((L.SFPU_full_mean * w).sum(), 2),
            "fpu_points": {g: round((x.FPU_act_mean * x.kern_ns).sum() / T, 2) for g, x in L.groupby("group")},
            "sfpu_points": {g: round((x.SFPU_act_mean * x.kern_ns).sum() / T, 2) for g, x in L.groupby("group")},
            "time_pct": {g: round(x.kern_ns.sum() / T * 100, 2) for g, x in L.groupby("group")},
        }
        res.append(r)
        print(
            f"{seq:>6}  FPU {r['fpu']:6.2f}  SFPU {r['sfpu']:6.2f}  either {r['either']:6.2f}  |  FPU pts {r['fpu_points']}  SFPU pts {r['sfpu_points']}"
        )
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
