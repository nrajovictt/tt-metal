#!/usr/bin/env python3
"""How far could the FPU clock drop in the Quasar two-thread SDPA model before a chunk step gets longer?

Stretches the FPU-thread tasks by a factor k and finds the largest k for which the step period stays within 0.1%.
Two bounds for "what slows with the FPU clock":
  whole task - the FPU thread's whole task time stretches (its waits on unpack/pack too)
  busy only  - only the FPU-busy cycles stretch; waits stay as measured
Lowest FPU clock with no slowdown = 1/k of nominal. Scenarios: today's split and matmul-only split
(sfpu_offload_model.py), each with the measured exp and the single-cycle exp. 256-token chunks (2k and up).

usage: python tools/tracy/prefill_study/fpu_clock_slack.py SPLIT.json [OUT.json]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quasar_model import simulate
from sfpu_offload_model import simulate_matmul_only


def largest_k(period_of):
    p0 = period_of(1.0)
    lo, hi = 1.0, 10.0
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if period_of(m) <= p0 * 1.001 else (lo, m)
    return p0, lo


def main():
    S = json.load(open(sys.argv[1]))["split_256"]
    busy = {"QK": S["QK"]["fpu"], "RED": S["MAX"]["fpu"], "PV": S["PV"]["fpu"], "CF": S["CORR"]["fpu"]}
    res = []
    for exp_name, exp_cyc in (("measured exp", S["SUBEXP"]["sfpu"]), ("single-cycle exp", 64 * 128)):
        se = S["SUBEXP"]["cycles"] - S["SUBEXP"]["sfpu"] + exp_cyc
        today = {
            "QK": S["QK"]["cycles"],
            "RED": S["MAX"]["cycles"] - S["MAX"]["sfpu"],
            "PV": S["PV"]["cycles"],
            "CF": S["CORR"]["cycles"] - S["CORR"]["sfpu"],
            "MS": S["MAX"]["sfpu"],
            "CE": S["CORR"]["sfpu"],
            "EX": se,
        }
        for bound in ("whole task", "busy only"):

            def period(k, c=today, bound=bound):
                cc = dict(c)
                for t in busy:
                    cc[t] = round(c[t] + busy[t] * (k - 1)) if bound == "busy only" else round(c[t] * k)
                return simulate(cc)[0]

            p0, k = largest_k(period)
            res.append(
                {
                    "exp": exp_name,
                    "split": "today's",
                    "bound": bound,
                    "step": p0,
                    "k": round(k, 3),
                    "clock_pct": round(100 / k, 1),
                }
            )
        mm = {
            "QK": S["QK"]["cycles"],
            "PV": S["PV"]["cycles"],
            "MAX": S["MAX"]["cycles"],
            "EX": se,
            "CO": S["CORR"]["cycles"],
        }
        p0, k = largest_k(lambda k: simulate_matmul_only(dict(mm, QK=round(mm["QK"] * k), PV=round(mm["PV"] * k)))[0])
        res.append(
            {
                "exp": exp_name,
                "split": "matmuls only (same cost)",
                "bound": "whole task",
                "step": p0,
                "k": round(k, 3),
                "clock_pct": round(100 / k, 1),
            }
        )
    for r in res:
        print(
            f"{r['exp']:<17}{r['split']:<26}{r['bound']:<11} step {r['step']:>7} cycles  FPU {r['k']:.2f}x slower -> {r['clock_pct']:5.1f}% clock"
        )
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
