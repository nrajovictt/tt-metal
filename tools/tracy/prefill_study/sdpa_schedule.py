#!/usr/bin/env python3
"""Measured SDPA schedule across the three compute threads (unpack / math / pack) from a zone run.

Prints, over all cores: the K-chunk step period (math QK start to next QK start), the gap from the end of a correction
to the next QK, and how far the unpack thread finishes sub+exp ahead of math. Also writes a two-step window of one core
(relative cycles) for the report's timeline.

usage: python tools/tracy/prefill_study/sdpa_schedule.py OUT.json generated/profiler/sdpa_zones/prefill_4k [STEP=4]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdpa_stages import load_sdpa_zones, zone_spans


def main():
    out, run = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d, rep, freq, kern_us = load_sdpa_zones(run)
    th = {r: zone_spans(d, r) for r in ("TRISC_0", "TRISC_1", "TRISC_2")}
    m, u = th["TRISC_1"], th["TRISC_0"]
    period, gap, lead = [], [], []
    for core, g in m.groupby("core"):
        qk = g[g.zone == "SDPA-QK"].t0.values
        dq = np.diff(qk)
        period += list(dq[(dq > 0.5 * np.median(dq)) & (dq < 2 * np.median(dq))])
        for t1 in g[g.zone == "SDPA-CORR"].t1.values:
            nxt = qk[qk > t1]
            if len(nxt):
                gap.append(nxt[0] - t1)
        ue = u[(u.core == core) & (u.zone == "SDPA-SUBEXP")].t1.values
        me = g[g.zone == "SDPA-SUBEXP"].t1.values
        n = min(len(ue), len(me))
        lead += list(me[:n] - ue[:n])
    core0 = sorted(m.core.unique())[0]
    qk0 = m[(m.core == core0) & (m.zone == "SDPA-QK")].t0.values
    base, stop = qk0[step], qk0[step + 2] + (qk0[step + 2] - qk0[step + 1])
    window = {
        r: [
            [z[5:], int(a - base), int(b - base)]
            for _, z, a, b in t[t.core == core0].itertuples(index=False)
            if b > base and a < stop
        ]
        for r, t in th.items()
    }
    res = {
        "run": run,
        "report": rep.name,
        "freq_mhz": freq,
        "core": list(map(int, core0)),
        "step_period_cycles": {
            "median": float(np.median(period)),
            "p5": float(np.percentile(period, 5)),
            "p95": float(np.percentile(period, 95)),
            "n": len(period),
        },
        "corr_to_next_qk_cycles_median": float(np.median(gap)),
        "math_minus_unpack_subexp_end_cycles_median": float(np.median(lead)),
        "window": window,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "window"}, indent=1))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
