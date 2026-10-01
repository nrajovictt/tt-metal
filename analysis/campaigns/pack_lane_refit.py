#!/usr/bin/env python3
"""Refit the PACK wall lane from the packer hardware counters instead of from device walls.

Offline, no arguments, reads $DD and writes beside the captures, like the other reducers. Nothing is
re-measured: every number comes from the `<tag>_zon_mp_counters.csv` files the multipass leg of
campaign_t21.sh already wrote, plus `inner_iters_model` out of decomp_summary.csv.

Why. The model prices the PACK lane with constants fitted against device walls (`_PACK_CAUSAL` and
`_PACK_NONCAUSAL` in polaris ttsim/perf/roofline_sdpa.py, runbook section 9). Two counters measure
the packer directly and are never read: PACKER_BUSY (cycles the packer pipeline is busy) and
PACKER_DEST_READ_AVAILABLE / DEST_READ_GRANTED_0 (cycles it is reading DEST, 64 B each). This
script fits both against the tile geometry of the step and reports what the fit implies.

The form fitted is per k-chunk step, on the mean core:

    counter_per_step = qct * (A + B * kct)        qct = q_chunk/32, kct = k_chunk/32

A is the per-q-tile cost that does not scale with k_chunk (the PV output tiles plus the running
softmax statistics); B is the per-(q-tile, k-tile) cost (the score/probability tile). Dividing each
by the ideal tile cost `tile_bytes / packer_bw_bytes_per_cycle` gives the implied tile counts, which
is what makes the result extrapolable to another head_dim or accumulate dtype.

Outputs: $DD/pack_lane_refit.csv, one row per config, plus the fitted constants on stdout.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path contract (PORTABLE_CONTRACT.md): DD holds the captures, POLARIS is only needed for the
# comparison against the constants currently in the model, and is optional.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))
OUT = DD / "pack_lane_refit.csv"

TAG = re.compile(r"^(t21_(causal|noncausal)_q(\d+)k(\d+))_zon_mp_counters\.csv$")

TILE_HW = 32
PACKER_BW_BYTES_PER_CYCLE = 64.0  # PACK_INTF_WIDTH(4) x DATA_WIDTH(128 bit), tt_tdma.sv:54
ACCUM_BYTES = 2  # bfloat16 DEST, the accumulate dtype of this grid

try:
    sys.path.insert(0, str(POLARIS / "ttsim" / "perf"))
    from roofline_sdpa import WALL_TERMS_BH  # noqa: E402

    HAVE_MODEL = True
except ModuleNotFoundError:
    HAVE_MODEL = False


def model_lane(causal, qct, kct, dct_sum):
    """The PACK lane the model prices today, cycles per k-chunk step. None without polaris."""
    if not HAVE_MODEL:
        return None
    t = WALL_TERMS_BH["prefill_causal" if causal else "prefill_noncausal"]
    return t.pack_per_step + qct * (t.pack_per_qtile + (t.pack_per_qktile + t.pack_per_qk_dtile * dct_sum) * kct)


def steps_per_config():
    """{tag: inner_iters} from decomp_summary.csv, the mean-core k-chunk step count."""
    path = DD / "decomp_summary.csv"
    if not path.exists():
        sys.exit(f"need {path}; run zone_decomp.py over the grid first")
    d = pd.read_csv(path)
    return dict(zip(d.tag, d.inner_iters_model))


def counter_means(path):
    """Mean over cores of the packer counters, last invocation (0 is a warmup)."""
    c = pd.read_csv(path)
    c = c[c.run_id == c.run_id.max()]
    s = c.groupby("counter").value.mean()
    return {
        "PACKER_BUSY": s.get("PACKER_BUSY", float("nan")),
        "PACKER_DEST_READ_AVAILABLE": s.get("PACKER_DEST_READ_AVAILABLE", float("nan")),
        "DEST_READ_GRANTED_0": s.get("DEST_READ_GRANTED_0", float("nan")),
    }


def card():
    for tag in sorted(DD.glob("t21_*_zoff.csv")):
        head = tag.open().readline()
        if head.startswith("PROVENANCE:"):
            return head[len("PROVENANCE:") :].split(";")[0].strip()
    return "unknown-card"


def fit(df, column):
    """Least squares for per_step = qct * (A + B * kct). Returns (A, B, percent residuals)."""
    x = np.c_[df.qct, df.qct * df.kct]
    (a, b), *_ = np.linalg.lstsq(x, df[column], rcond=None)
    resid = 100 * (x @ np.array([a, b]) - df[column]) / df[column]
    return a, b, resid


def collect(steps):
    rows = []
    for path in sorted(DD.glob("t21_*_zon_mp_counters.csv")):
        m = TAG.match(path.name)
        if not m:
            continue
        tag, mode, q, k = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if tag not in steps:
            continue
        ctr = counter_means(path)
        if not np.isfinite(ctr["PACKER_BUSY"]) or not np.isfinite(ctr["DEST_READ_GRANTED_0"]):
            print(f"skipping {tag}: packer counters absent from the capture")
            continue
        n = steps[tag]
        rows.append(
            dict(
                tag=tag,
                causal=int(mode == "causal"),
                q_chunk=q,
                k_chunk=k,
                qct=q // TILE_HW,
                kct=k // TILE_HW,
                steps=n,
                packer_busy=ctr["PACKER_BUSY"],
                dest_read=ctr["DEST_READ_GRANTED_0"],
                dest_read_avail=ctr["PACKER_DEST_READ_AVAILABLE"],
                packer_busy_per_step=ctr["PACKER_BUSY"] / n,
                dest_read_per_step=ctr["DEST_READ_GRANTED_0"] / n,
            )
        )
    return pd.DataFrame(rows)


def main():
    df = collect(steps_per_config())
    if df.empty:
        sys.exit(f"no usable t21_*_zon_mp_counters.csv under {DD}")

    # head_dim is fixed at 128 across the T2.1 grid, so dct_v and dct_sum cannot be separated here.
    dct_v, dct_sum = 4, 8
    ideal_tile_cycles = TILE_HW * TILE_HW * ACCUM_BYTES / PACKER_BW_BYTES_PER_CYCLE

    a_dr, b_dr, resid_dr = fit(df, "dest_read_per_step")
    a_pb, b_pb, resid_pb = fit(df, "packer_busy_per_step")
    df["dest_read_fit_pct"] = resid_dr
    df["packer_busy_fit_pct"] = resid_pb
    df["occupancy_factor"] = df.packer_busy / df.dest_read
    df["dest_read_backpressure"] = df.dest_read_avail - df.dest_read
    df["model_lane_per_step"] = [
        model_lane(bool(r.causal), r.qct, r.kct, dct_sum) for r in df.itertuples()
    ]
    if HAVE_MODEL:
        df["model_lane_over_packer_busy"] = df.model_lane_per_step / df.packer_busy_per_step

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_SD).stdout.strip()
    cols = [c for c in df.columns if c != "tag"]
    with OUT.open("w", newline="") as fh:
        fh.write(
            f"# PROVENANCE: {card()}; tt-metal {sha}; packer counters from the zones-on multipass leg, "
            f"mean over cores of the last invocation; steps = inner_iters_model from decomp_summary.csv; "
            f"fit per_step = qct*(A + B*kct)\n"
        )
        w = csv.DictWriter(fh, fieldnames=["tag", *cols])
        w.writeheader()
        for r in df.to_dict("records"):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{len(df)} configs from {DD}\n")
    print("Fitted per-step form, counters vs tile geometry")
    print(f"  DEST_READ_GRANTED_0 = qct * ({a_dr:7.1f} + {b_dr:6.2f} * kct)"
          f"   mean |resid| {np.abs(resid_dr).mean():.1f}%  max {np.abs(resid_dr).max():.1f}%")
    print(f"  PACKER_BUSY         = qct * ({a_pb:7.1f} + {b_pb:6.2f} * kct)"
          f"   mean |resid| {np.abs(resid_pb).mean():.1f}%  max {np.abs(resid_pb).max():.1f}%")

    print(f"\nImplied tile counts at {ideal_tile_cycles:.0f} cycles per {ACCUM_BYTES}-byte tile "
          f"({PACKER_BW_BYTES_PER_CYCLE:.0f} B/cycle packer)")
    print(f"  per q-tile per step   {a_dr / ideal_tile_cycles:5.2f} tiles   "
          f"(dct_v + 1 = {dct_v + 1} would be output tiles plus the softmax statistics)")
    print(f"  per q-k-tile per step {b_dr / ideal_tile_cycles:5.2f} tiles   "
          f"(the score/probability tile)")

    occ = df.occupancy_factor
    print(f"\nPacker occupancy factor PACKER_BUSY / DEST_READ_GRANTED_0")
    print(f"  mean {occ.mean():.3f}  min {occ.min():.3f}  max {occ.max():.3f}  spread {100 * (occ.max() / occ.min() - 1):.1f}%")

    bp = df.dest_read_backpressure
    print(f"\nDEST read back-pressure, PACKER_DEST_READ_AVAILABLE - DEST_READ_GRANTED_0")
    print(f"  max {bp.abs().max():.0f} cycles over {len(df)} configs"
          f"{'  (the pack request FIFO never fills)' if bp.abs().max() == 0 else ''}")

    print("\nCausality, matched (q_chunk, k_chunk), DEST reads per step")
    for (q, k), g in df.groupby(["q_chunk", "k_chunk"]):
        if g.causal.nunique() != 2:
            continue
        c = g[g.causal == 1].dest_read_per_step.iloc[0]
        n = g[g.causal == 0].dest_read_per_step.iloc[0]
        print(f"  q{q:<4} k{k:<4} causal {c:9.1f}   non-causal {n:9.1f}   {100 * (n / c - 1):+6.2f}%")

    if HAVE_MODEL:
        r = df.model_lane_over_packer_busy
        print(f"\nModel PACK lane against measured packer occupancy")
        print(f"  lane / PACKER_BUSY  mean {r.mean():.2f}  min {r.min():.2f}  max {r.max():.2f}")
        print(f"  so roughly {100 / r.mean():.0f}% of the lane is packer hardware and the rest is pack-thread software")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
