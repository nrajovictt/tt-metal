#!/usr/bin/env python3
"""Measure whole-op DRAM traffic from the L1 NOC port counters and join it to the model.

Offline, no arguments, reads $DD and writes beside the captures, like the other reducers. Nothing is
re-measured: the numbers come from the `<tag>_zon_mp_counters.csv` files the multipass leg of
campaign_t21.sh already wrote.

Every L1 client port carries a request and a grant counter. A grant is one accepted access of
DATA_WIDTH = 128 bit, so **bytes = grants x 16** (tt_tensix_with_l1.sv:25, and the L1 address is
16-byte-word addressed). Each NOC ring exposes four independent channels and the traffic stripes
evenly across them, so all four must be summed.

The direction names are from the NIU's point of view, not L1's, which is the opposite of what the
names suggest. Established empirically rather than from the RTL: summing the ring0 *outgoing* grants
reproduces the analytic Q + K/V re-read volume to 0.00 percent on all seven causal points of the
T2.1 grid, so "outgoing" is data leaving the NIU into L1, i.e. a DRAM read landing locally. Ring1
incoming then carries the output written back, and matches the model's dram_out_bytes to 3 percent.

Why it matters: the model's `dram_in_bytes` is a read *footprint* (each tensor counted once) and is
constant across the grid, while the traffic that actually crosses the NOC re-reads K and V once per
visiting q chunk and so varies by more than 10x with q_chunk. The model gained a separate
`dram_read_traffic_bytes` for that; this script is what validates it.

Outputs: $DD/noc_bytes.csv, one row per config, plus a summary on stdout.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Path contract (PORTABLE_CONTRACT.md). POLARIS is only needed for the model columns and is optional.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))
OUT = DD / "noc_bytes.csv"

TAG = re.compile(r"^(t21_(causal|noncausal)_q(\d+)k(\d+))_zon_mp_counters\.csv$")

BYTES_PER_GRANT = 16  # DATA_WIDTH 128 bit at every L1 client port

# Blackhole exposes two NOC rings of four channels each, split over L1 counter banks 0/2 (ring0) and
# 1/3 (ring1). See hw_counters.h; the bank split is only the count-time mux, not a topology boundary.
# Listed by *grant* counter name rather than derived from the request name: L1 ports 1 and 8 carry
# generically named grants (L1_0_PORT1_GRANT, L1_1_PORT8_GRANT) because they sit behind a round-robin
# arbiter shared with another client, so "<request>_GRANT" silently misses them and undercounts.
RING0_INTO_L1 = [
    "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    "L1_2_NOC_RING0_OUTGOING_2_GRANT",
    "L1_2_NOC_RING0_OUTGOING_3_GRANT",
]
RING0_OUT_OF_L1 = [
    "L1_0_NOC_RING0_INCOMING_0_GRANT",
    "L1_0_NOC_RING0_INCOMING_1_GRANT",
    "L1_2_NOC_RING0_INCOMING_2_GRANT",
    "L1_2_NOC_RING0_INCOMING_3_GRANT",
]
RING1_OUT_OF_L1 = [
    "L1_1_NOC_RING1_INCOMING_0_GRANT",
    "L1_1_NOC_RING1_INCOMING_1_GRANT",
    "L1_3_NOC_RING1_INCOMING_2_GRANT",
    "L1_3_NOC_RING1_INCOMING_3_GRANT",
]
RING1_INTO_L1 = [
    "L1_1_NOC_RING1_OUTGOING_0_GRANT",
    "L1_1_NOC_RING1_OUTGOING_1_GRANT",
    "L1_3_NOC_RING1_OUTGOING_2_GRANT",
    "L1_3_NOC_RING1_OUTGOING_3_GRANT",
]
# On-chip client ports: L1 into SrcA/SrcB, and DEST out to L1. Per core, the basis the model's
# unpack_bytes_total / pack_bytes_total use.
#
# Calibration caveat, established 2026-09-17: summing the eight packer ports at 16 bytes a grant gives
# 2.006x (spread 1.96 to 2.11, which is the model's own 1.4 percent fit residual doubled) the bytes that
# cross the DEST read interface, on all 27 board-config pairs. The packer cannot write twice what it
# reads -- the output format is never wider than DEST -- so the group double counts: four of the eight
# ports carry the same data as the other four. The honest conversion for this group is therefore
# grants x 8. It is left at x16 below so the raw measurement stays visible; divide before comparing
# against a bytes-per-cycle bound. The unpacker group ratio is not constant (1.12 to 2.85), so it has
# either a different aliasing factor or a residual shape error, and is not calibrated.
UNPACK_PORTS = [
    "L1_0_UNPACKER_0_GRANT",
    "L1_0_PORT1_GRANT",
    "L1_1_EXT_UNPACKER_1_GRANT",
    "L1_1_EXT_UNPACKER_2_GRANT",
    "L1_1_EXT_UNPACKER_3_GRANT",
    "L1_2_EXT_UNPACKER_4_GRANT",
    "L1_2_EXT_UNPACKER_5_GRANT",
    "L1_2_EXT_UNPACKER_6_GRANT",
    "L1_2_EXT_UNPACKER_7_GRANT",
]
PACK_PORTS = [
    "L1_1_PORT8_GRANT",
    "L1_3_TDMA_PACK_EXT_0_GRANT",
    "L1_3_TDMA_PACK_EXT_1_GRANT",
    "L1_3_TDMA_PACK_EXT_2_GRANT",
    "L1_3_TDMA_PACK_EXT_3_GRANT",
    "L1_4_TDMA_PACK_EXT_4_GRANT",
    "L1_4_TDMA_PACK_EXT_5_GRANT",
    "L1_4_TAG_SEARCH_PACKER1_GRANT",
]

try:
    sys.path.insert(0, str(POLARIS / "ttsim" / "perf"))
    from roofline_sdpa import SdpaConfig, predict  # noqa: E402

    HAVE_MODEL = True
except ModuleNotFoundError:
    HAVE_MODEL = False


def config_from_provenance(tag):
    """S, head count and dtypes off the PROVENANCE line of the zones-on capture, as zone_decomp does."""
    path = DD / f"{tag}_zon.csv"
    prov = path.open().readline() if path.exists() else ""

    def env(key, default):
        m = re.search(r"\b" + key + r"=(\S+)", prov)
        return m.group(1) if m else default

    return int(env("S", 4096)), int(env("nh", 32)), int(env("nkv", 8)), int(env("d", 128))


def grant_bytes(counters, grant_names):
    """Bytes moved over a port group: grants summed over its channels and all cores."""
    return float(counters[counters.counter.isin(grant_names)].value.sum()) * BYTES_PER_GRANT


def card():
    for tag in sorted(DD.glob("t21_*_zoff.csv")):
        head = tag.open().readline()
        if head.startswith("PROVENANCE:"):
            return head[len("PROVENANCE:") :].split(";")[0].strip()
    return "unknown-card"


def collect():
    rows = []
    for path in sorted(DD.glob("t21_*_zon_mp_counters.csv")):
        m = TAG.match(path.name)
        if not m:
            continue
        tag, mode, q, k = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        c = pd.read_csv(path)
        c = c[c.run_id == c.run_id.max()]  # one invocation; 0 is a warmup
        read_in = grant_bytes(c, RING0_INTO_L1)
        if read_in == 0:
            print(f"skipping {tag}: no NOC grants in the capture")
            continue
        S, nh, nkv, d = config_from_provenance(tag)
        row = dict(
            tag=tag,
            causal=int(mode == "causal"),
            q_chunk=q,
            k_chunk=k,
            cores=int(c[["core_x", "core_y"]].drop_duplicates().shape[0]),
            read_bytes_into_l1=read_in,
            forward_bytes_out_of_l1=grant_bytes(c, RING0_OUT_OF_L1),
            write_bytes_out_of_l1=grant_bytes(c, RING1_OUT_OF_L1),
            ring1_bytes_into_l1=grant_bytes(c, RING1_INTO_L1),
        )
        cores = row["cores"]
        row["unpack_bytes_per_core"] = grant_bytes(c, UNPACK_PORTS) / cores
        row["pack_bytes_per_core"] = grant_bytes(c, PACK_PORTS) / cores
        if HAVE_MODEL:
            r = predict(
                SdpaConfig(
                    S=S, head_dim=d, q_chunk=q, k_chunk=k, num_heads=nh, num_kv_heads=nkv,
                    num_cores=row["cores"], is_causal=bool(row["causal"]), fidelity="HiFi2",
                )
            )
            row["model_footprint_bytes"] = r.dram_in_bytes
            row["model_read_traffic_bytes"] = getattr(r, "dram_read_traffic_bytes", float("nan"))
            row["model_write_bytes"] = r.dram_out_bytes
            # A capture can come back with a direction entirely empty (seen on non-causal q512 k512,
            # the same tag whose packer counters are missing); leave the error blank rather than divide.
            wrote = row["write_bytes_out_of_l1"]
            row["read_traffic_err_pct"] = 100 * (row["model_read_traffic_bytes"] / read_in - 1)
            row["write_err_pct"] = 100 * (r.dram_out_bytes / wrote - 1) if wrote else float("nan")
            row["traffic_over_footprint"] = read_in / r.dram_in_bytes
            # On-chip: the model's unpack/pack totals count operand tiles once and omit the per-step
            # score/probability round trip through L1, so both read low. Ratios, not errors.
            row["model_unpack_bytes_per_core"] = r.unpack_bytes_total
            row["model_pack_bytes_per_core"] = r.pack_bytes_total
            row["unpack_measured_over_model"] = row["unpack_bytes_per_core"] / r.unpack_bytes_total
            row["pack_measured_over_model"] = (
                row["pack_bytes_per_core"] / r.pack_bytes_total if r.pack_bytes_total else float("nan")
            )
            # What the measured bytes imply as an L1 bandwidth floor, against the compute floor it
            # would have to beat to bind. Never binds on this grid, but the model's own floor is a
            # factor of 20 further away, so it could not tell you when it starts to matter.
            row["l1_floor_cycles"] = max(row["unpack_bytes_per_core"] / 80.0,
                                         row["pack_bytes_per_core"] / 64.0)
            row["l1_floor_over_math_active"] = row["l1_floor_cycles"] / r.math_active_cycles
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = collect()
    if df.empty:
        sys.exit(f"no usable t21_*_zon_mp_counters.csv under {DD}")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_SD).stdout.strip()
    with OUT.open("w", newline="") as fh:
        fh.write(
            f"# PROVENANCE: {card()}; tt-metal {sha}; L1 NOC grant counters from the zones-on multipass leg, "
            f"summed over all cores of the last invocation, {BYTES_PER_GRANT} bytes per grant, "
            f"four channels per ring direction\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(df.columns))
        w.writeheader()
        for r in df.to_dict("records"):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    mb = 1e6
    have = HAVE_MODEL and "model_read_traffic_bytes" in df
    head = f"{'config':22s}{'read MB':>10s}{'write MB':>10s}{'fwd MB':>9s}"
    if have:
        head += f"{'model read MB':>15s}{'err':>8s}{'vs footprint':>14s}"
    print(f"{len(df)} configs from {DD}\n{head}")
    for r in df.itertuples():
        line = (f"{('causal' if r.causal else 'non-causal') + ' q' + str(r.q_chunk) + ' k' + str(r.k_chunk):22s}"
                f"{r.read_bytes_into_l1 / mb:10.1f}{r.write_bytes_out_of_l1 / mb:10.1f}"
                f"{r.forward_bytes_out_of_l1 / mb:9.1f}")
        if have:
            line += (f"{r.model_read_traffic_bytes / mb:15.1f}{r.read_traffic_err_pct:7.2f}%"
                     f"{r.traffic_over_footprint:13.1f}x")
        print(line)

    if have:
        good = df[df.read_traffic_err_pct.abs() < 10]
        print(f"\nread traffic vs model, excluding {len(df) - len(good)} outlier(s) over 10 percent:"
              f"  mean |err| {good.read_traffic_err_pct.abs().mean():.2f}%"
              f"  max {good.read_traffic_err_pct.abs().max():.2f}%")
        print(f"write traffic vs model dram_out_bytes:"
              f"  mean |err| {df.write_err_pct.abs().mean():.2f}%"
              f"  max {df.write_err_pct.abs().max():.2f}%")
        print(f"actual read traffic is {df.traffic_over_footprint.min():.1f}x to "
              f"{df.traffic_over_footprint.max():.1f}x the model's dram_in_bytes footprint")

        print(f"\n{'config':22s}{'unpack MB':>11s}{'vs model':>10s}{'pack MB':>10s}{'vs model':>10s}"
              f"{'L1 floor':>10s}{'/ math':>8s}")
        for r in df.itertuples():
            print(f"{('causal' if r.causal else 'non-causal') + ' q' + str(r.q_chunk) + ' k' + str(r.k_chunk):22s}"
                  f"{r.unpack_bytes_per_core / mb:11.2f}{r.unpack_measured_over_model:9.1f}x"
                  f"{r.pack_bytes_per_core / mb:10.2f}{r.pack_measured_over_model:9.1f}x"
                  f"{r.l1_floor_cycles / 1e3:9.0f}k{r.l1_floor_over_math_active:8.2f}")
        onchip = df[df.pack_bytes_per_core > 0]
        print(f"\non-chip L1 traffic per core, measured against the model:"
              f"  unpack {onchip.unpack_measured_over_model.min():.1f}x to "
              f"{onchip.unpack_measured_over_model.max():.1f}x,"
              f"  pack {onchip.pack_measured_over_model.min():.0f}x to "
              f"{onchip.pack_measured_over_model.max():.0f}x")
        print(f"implied L1 bandwidth floor reaches {onchip.l1_floor_over_math_active.max():.2f} of the "
              f"compute floor, so it never binds on this grid")
        pr = onchip.pack_measured_over_model
        print(f"packer port group reads {pr.mean():.3f}x the model, i.e. the eight ports double count "
              f"the DEST-read basis; halve before using as a bandwidth bound (see the note on PACK_PORTS)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
