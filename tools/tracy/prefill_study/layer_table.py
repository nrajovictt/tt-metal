#!/usr/bin/env python3
"""Corrected per-length table: layer kernel time, MatMul % / SDPA %, FPU-in-matmul and SFPU-in-SDPA utilization.

Scope "layer" = Embeddings + one decoder layer: the Embeddings op plus every op from the first LayerNorm to the
last BinaryNg (the MLP residual add). This is the same set of ops whether prefill was traced (128, 1k) or eager,
and it leaves out the final norm, the LM head matmuls and sampling. Scope "full" = the whole last forward pass.

Utilization (same definitions as perf_util_report.py, averaged over ops weighted by kernel time):
  act  = median over the cores that ran the op of counter / that core's counter window
  full = mean over all worker-grid cores (idle cores = 0) of counter / op kernel duration

usage:
  python tools/tracy/prefill_study/layer_table.py OUT.json 128:generated/profiler/prefill_128_1dev 4096:generated/profiler/prefill_4k_1dev ...
Calls perf_util_report.py --select last-forward --csv for each run (in parallel), keeps the per-op CSVs next to OUT.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1] / "perf_util_report.py"


def wavg(v, w):
    ok = v.notna() & w.notna()
    return round(float(np.average(v[ok], weights=w[ok])), 2) if ok.any() else None


def stats(d):
    t = d.kern_ns.sum()
    mm, sd = d[d.kind == "matmul"], d[d.kind == "sdpa"]
    return {
        "ops": int(len(d)),
        "kernel_us": round(t / 1e3, 1),
        "matmul_us": round(mm.kern_ns.sum() / 1e3, 1),
        "sdpa_us": round(sd.kern_ns.sum() / 1e3, 1),
        "matmul_pct": round(mm.kern_ns.sum() / t * 100, 2),
        "sdpa_pct": round(sd.kern_ns.sum() / t * 100, 2),
        "fpu_matmul_act": wavg(mm.FPU_act_med, mm.kern_ns),
        "fpu_matmul_full": wavg(mm.FPU_full_mean, mm.kern_ns),
        "sfpu_sdpa_act": wavg(sd.SFPU_act_med, sd.kern_ns),
        "sfpu_sdpa_work": wavg(sd.SFPU_work_med, sd.kern_ns),
        "sfpu_sdpa_full": wavg(sd.SFPU_full_mean, sd.kern_ns),
        "fpu_sdpa_act": wavg(sd.FPU_act_med, sd.kern_ns),
    }


def layer_scope(full):
    a = full.index[full.code == "LayerNorm"][0]
    b = full.index[full.code == "BinaryNg"][-1]
    return pd.concat([full[full.code == "Embeddings"], full.loc[a:b]])


def main():
    out = Path(sys.argv[1])
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    runs = [(int(a.split(":", 1)[0]), a.split(":", 1)[1]) for a in sys.argv[2:]]
    procs = []
    for seq, run in runs:
        csv = work / f"ops_{seq}.csv"
        log = open(work / f"report_{seq}.txt", "w")
        procs.append(
            (
                seq,
                run,
                csv,
                subprocess.Popen(
                    [sys.executable, str(REPORT), run, "--select", "last-forward", "--csv", str(csv)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    res = []
    for seq, run, csv, p in procs:
        if p.wait() != 0:
            sys.exit(f"[error] perf_util_report failed for {run}, see {work}/report_{seq}.txt")
        full = pd.read_csv(csv, low_memory=False)
        layer = layer_scope(full)
        res.append(
            {
                "seq": seq,
                "run": run,
                "layer": stats(layer),
                "full": stats(full),
                "layer_ops": [f"{c}:{s}" for c, s in zip(layer.code, layer["shape"])],
                "layer_rows": [int(layer.idx.min()), int(layer.idx.max())],
            }
        )
        L = res[-1]["layer"]
        print(
            f"{seq:>6}  layer {L['kernel_us']:>10.1f} us  ops {L['ops']:>2}  MM {L['matmul_pct']:6.2f}%  SDPA {L['sdpa_pct']:6.2f}%  "
            f"FPU act {L['fpu_matmul_act']}  full {L['fpu_matmul_full']}  SFPU act {L['sfpu_sdpa_act']}  full {L['sfpu_sdpa_full']}  "
            f"rows {res[-1]['layer_rows']}"
        )
    res.sort(key=lambda r: r["seq"])
    out.write_text(json.dumps(res, indent=1))
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
