#!/usr/bin/env python3
"""Check how the polaris SDPA roofline fits the measured device wall, and localise the residual.

Offline, no arguments, no card. Reads $DD and writes beside the captures, like the other reducers.

There are two ways to check the model against a wall. The one polaris ships,
`$POLARIS/ttsim/perf/sdpa_validate.py`, reads a tracy `ops_perf_results_*.csv` and compares
`DEVICE KERNEL DURATION [ns]` per op; it is the acceptance path (runbook 9.5) but it needs a run made
with `-r` and *without* `--perf-counter-multipass`, which the T2.1 campaign did not produce. This
script is the other way: it uses the zones-off device walls the campaign already captured, so it runs
on data in hand, and it goes further than a pass/fail by attributing the residual.

Attribution is possible because the model's wall is a sum of named lanes and two of its inputs can be
checked independently against the same captures:

  * the work split -- `steps_wall_core` and `q_chunks_wall_core` against the zone data's
    `it1_steps_wc` / `it1_qchunks_wc`. On the T2.1 grid this is exact on 14 of 14, which matters: it
    means a wall residual is never the model mis-counting work, only mis-pricing a step.
  * the compute floor -- `math_active_cycles` against `MATH_COUNTER`.

With the work split exact, the residual divided by the wall core's step count is a per-step price
error, and printing it beside each lane's per-step cost shows which lane could absorb it. A residual
that is a large fraction of one lane and a rounding error on the others points at that lane.

Outputs: $DD/wall_fit.csv, one row per config, plus the tables on stdout.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))
OUT = DD / "wall_fit.csv"

TAG = re.compile(r"^t21_(causal|noncausal)_q(\d+)k(\d+)$")

sys.path.insert(0, str(POLARIS / "ttsim" / "perf"))
try:
    from roofline_sdpa import SdpaConfig, predict  # noqa: E402
except ModuleNotFoundError:
    sys.exit(f"need the model at {POLARIS}/ttsim/perf/roofline_sdpa.py; set POLARIS to override")

# Lanes the model prices per k-chunk step. init and straggler are whole-op, so they are handled apart.
PER_STEP_LANES = ["compute_floor", "fe_issue", "reader_wait", "mask_bracket", "sfpu_issue",
                  "dest_roundtrip", "control"]


def card():
    for t in sorted(DD.glob("t21_*_zoff.csv")):
        head = t.open().readline()
        if head.startswith("PROVENANCE:"):
            return head[len("PROVENANCE:"):].split(";")[0].strip()
    return "unknown-card"


def collect():
    path = DD / "decomp_summary.csv"
    if not path.exists():
        sys.exit(f"need {path}; run zone_decomp.py over the grid first")
    d = pd.read_csv(path).set_index("tag")
    rows = []
    for tag in d.index:
        m = TAG.match(str(tag))
        if not m:
            continue
        causal, q, k = m.group(1) == "causal", int(m.group(2)), int(m.group(3))
        rec = d.loc[tag]
        meas = rec.get("wall_zoff_mean")
        if pd.isna(meas):
            continue
        S = int(rec.get("S", 4096) or 4096)
        cores = int(rec.get("cores", 110) or 110)
        r = predict(SdpaConfig(S=S, head_dim=128, q_chunk=q, k_chunk=k, num_heads=32, num_kv_heads=8,
                               num_cores=cores, is_causal=causal,
                               fidelity=str(rec.get("fidelity", "HiFi2") or "HiFi2")))
        steps_meas = rec.get("it1_steps_wc")
        row = dict(
            tag=tag, causal=int(causal), q_chunk=q, k_chunk=k,
            wall_pred=r.wall_clock_cycles, wall_meas=meas,
            wall_err_pct=100 * (r.wall_clock_cycles / meas - 1),
            resid_cycles=r.wall_clock_cycles - meas,
            steps_model=r.steps_wall_core, steps_meas=steps_meas,
            steps_err_pct=(100 * (r.steps_wall_core / steps_meas - 1)) if steps_meas else float("nan"),
            qchunks_model=r.q_chunks_wall_core, qchunks_meas=rec.get("it1_qchunks_wc"),
            floor_model=r.math_active_cycles, floor_meas=rec.get("ctr_MATH_COUNTER_mean"),
        )
        row["floor_err_pct"] = (100 * (row["floor_model"] / row["floor_meas"] - 1)
                                if pd.notna(row["floor_meas"]) else float("nan"))
        steps = r.steps_wall_core or 1
        row["resid_per_step"] = row["resid_cycles"] / steps
        for lane in PER_STEP_LANES:
            row[f"lane_{lane}_per_step"] = r.components.get(lane, 0.0) / steps
        row["lane_init"] = r.components.get("init", 0.0)
        row["lane_straggler"] = r.components.get("straggler", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = collect()
    if df.empty:
        sys.exit(f"no t21_* rows with a zones-off wall under {DD}")
    df = df.sort_values(["causal", "q_chunk", "k_chunk"], ascending=[False, True, True])

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_SD).stdout.strip()
    with OUT.open("w", newline="") as fh:
        fh.write(f"# PROVENANCE: {card()}; tt-metal {sha}; model {POLARIS}/ttsim/perf/roofline_sdpa.py; "
                 f"measured = wall_zoff_mean (zones off, mean of invocations 1 and 2)\n")
        w = csv.DictWriter(fh, fieldnames=list(df.columns))
        w.writeheader()
        for rec in df.to_dict("records"):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})

    def label(r):
        return f"{'causal' if r.causal else 'non-causal'} q{r.q_chunk} k{r.k_chunk}"

    print(f"{len(df)} configs from {DD}\n")
    print("1. Wall fit")
    print(f"  {'config':22s}{'predicted':>12s}{'measured':>12s}{'error':>9s}{'residual':>12s}")
    for r in df.itertuples():
        print(f"  {label(r):22s}{r.wall_pred:12,.0f}{r.wall_meas:12,.0f}"
              f"{r.wall_err_pct:8.2f}%{r.resid_cycles:12,.0f}")
    print(f"\n  mean |error| {df.wall_err_pct.abs().mean():.2f}%   max {df.wall_err_pct.abs().max():.2f}%"
          f"   bias {df.wall_err_pct.mean():+.2f}%")

    print("\n2. Is the residual a work-split error? (model against the zone data)")
    bad = df[df.steps_err_pct.abs() > 0.05]
    if bad.empty:
        print(f"  no: steps_wall_core and q_chunks_wall_core match on all {len(df)} configs, exactly.")
        print("      So every residual below is a per-step price error, not a miscount of work.")
    else:
        for r in bad.itertuples():
            print(f"  {label(r):22s} steps model {r.steps_model:.0f} vs measured {r.steps_meas:.0f}"
                  f" ({r.steps_err_pct:+.1f}%) -- fix the work split before reading the lanes")

    have_floor = df.floor_err_pct.notna()
    if have_floor.any():
        f = df[have_floor]
        print("\n3. Is it the compute floor? (math_active_cycles against MATH_COUNTER)")
        print(f"  mean {f.floor_err_pct.mean():+.2f}%   range {f.floor_err_pct.min():+.2f}% to "
              f"{f.floor_err_pct.max():+.2f}%")

    print("\n4. Which lane could absorb the residual? (cycles per step on the wall core)")
    lanes = [c for c in df.columns if c.startswith("lane_") and c.endswith("_per_step")]
    live = [c for c in lanes if df[c].abs().max() > 0.5]
    hdr = "".join(f"{c[5:-9][:11]:>12s}" for c in live)
    print(f"  {'config':22s}{'resid/step':>12s}{hdr}")
    for r in df.itertuples():
        vals = "".join(f"{getattr(r, c):12,.0f}" for c in live)
        print(f"  {label(r):22s}{r.resid_per_step:12,.0f}{vals}")
    print("\n  A residual that is a few percent of one lane and larger than the others put together")
    print("  points at that lane's constants. Compare against pack_lane_refit.py and noc_bytes.py,")
    print("  which price the PACK and DRAM lanes directly from counters.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
