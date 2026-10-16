#!/usr/bin/env python3
"""Per-stage SDPA breakdown from the SDPA-* zones added by sdpa_stage_zones.patch.

Each TRISC records only its first ~125 zones per op (L1 profiler buffer), so the sample over-represents the first
iterations on each core. We take the mean duration of each stage from that sample and multiply it by the exact number
of times the stage runs in a causal SDPA with equal Q/K chunks (32 heads, nq = ceil(seq/chunk) chunks):
  per K-chunk step  QK, MAX, SUBEXP, PV : 32 * nq(nq+1)/2      CORR : that minus 32*nq (first step has no correction)
  per Q chunk       MASK, ROWSUM, NORM  : 32 * nq
Chunk = 256 tokens from 2k up, 64 below (model_config.get_attn_sdpa_prefill_program_config).
"coverage" = rebuilt stage total / measured TRISC kernel time summed over cores; ~96-100% means nothing is missing.

usage: python tools/tracy/prefill_study/sdpa_stages.py OUT.json 4096:generated/profiler/sdpa_zones/prefill_4k [...]
"""

import json
import sys
from pathlib import Path

import pandas as pd

HEADS = 32
STAGES = ["SDPA-QK", "SDPA-MASK", "SDPA-MAX", "SDPA-SUBEXP", "SDPA-PV", "SDPA-CORR", "SDPA-ROWSUM", "SDPA-NORM"]
RISCS = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}
USE = ["core_x", "core_y", "RISC processor type", "time[cycles since reset]", "run host ID", "zone name", "type"]


def stage_counts(seq):
    chunk = 256 if seq >= 2048 else 64
    nq = -(-seq // chunk)
    iters, per_q = HEADS * nq * (nq + 1) // 2, HEADS * nq
    return chunk, {
        "SDPA-QK": iters,
        "SDPA-MASK": per_q,
        "SDPA-MAX": iters,
        "SDPA-SUBEXP": iters,
        "SDPA-PV": iters,
        "SDPA-CORR": iters - per_q,
        "SDPA-ROWSUM": per_q,
        "SDPA-NORM": per_q,
    }


def load_sdpa_zones(run_dir):
    """Device-log rows of the last SDPA op in the run. Returns (rows, report_dir, freq_mhz, kernel_us)."""
    rep = sorted(d for d in Path(run_dir, "reports").iterdir() if d.is_dir())[-1]
    ops = pd.read_csv(next(rep.glob("ops_perf_results_*.csv")), low_memory=False)
    sd = ops[ops["OP CODE"].str.contains("SDPA") & ops["DEVICE KERNEL DURATION [ns]"].notna()].iloc[-1]
    call = int(sd["GLOBAL CALL COUNT"])
    dev_log = rep / "profile_log_device.csv"
    with open(dev_log) as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        cols = [c.strip() for c in f.readline().split(",")]
    it = pd.read_csv(dev_log, skiprows=2, header=None, names=cols, usecols=USE, chunksize=2_000_000, low_memory=False)
    d = pd.concat(c[c["run host ID"] == call] for c in it)
    d = d.rename(columns={"RISC processor type": "risc", "time[cycles since reset]": "ts", "zone name": "zone"})
    return d, rep, freq, float(sd["DEVICE KERNEL DURATION [ns]"]) / 1e3


def zone_spans(d, risc):
    """[(core, stage, start, end)] for the SDPA-* zones of one RISC, paired start/end in time order per core."""
    out = []
    for core, g in d[(d.risc == risc) & d.zone.str.startswith("SDPA-", na=False)].groupby(["core_x", "core_y"]):
        g = g.sort_values("ts")
        s, e = g[g.type == "ZONE_START"], g[g.type == "ZONE_END"]
        assert len(s) == len(e) and (s.zone.values == e.zone.values).all(), f"unpaired zones on {core} {risc}"
        out += [(core, z, a, b) for z, a, b in zip(s.zone.values, s.ts.values, e.ts.values)]
    return pd.DataFrame(out, columns=["core", "zone", "t0", "t1"])


def analyse(seq, run_dir):
    d, rep, freq, kern_us = load_sdpa_zones(run_dir)
    chunk, counts = stage_counts(seq)
    out = {
        "seq": seq,
        "run": str(run_dir),
        "report": rep.name,
        "chunk": chunk,
        "freq_mhz": freq,
        "sdpa_kernel_us": kern_us,
        "counts": counts,
        "riscs": {},
    }
    for risc, role in RISCS.items():
        k = (
            d[(d.risc == risc) & (d.zone == "TRISC-KERNEL")]
            .groupby(["core_x", "core_y"])
            .ts.agg(lambda s: s.max() - s.min())
        )
        kernel_total_us = float(k.sum()) / freq
        z = zone_spans(d, risc)
        z["us"] = (z.t1 - z.t0) / freq
        st = z.groupby("zone").us.agg(["mean", "median", "count"])
        stages, total = {}, 0.0
        for s in STAGES:
            if s in st.index:
                tot = float(st.loc[s, "mean"]) * counts[s]
                total += tot
                stages[s] = {
                    "mean_us": float(st.loc[s, "mean"]),
                    "median_us": float(st.loc[s, "median"]),
                    "samples": int(st.loc[s, "count"]),
                    "est_total_us": tot,
                }
        for v in stages.values():
            v["share_pct"] = v["est_total_us"] / total * 100
        out["riscs"][role] = {
            "cores": int((k > 0).sum()),
            "kernel_total_us": kernel_total_us,
            "coverage_pct": total / kernel_total_us * 100,
            "stages": stages,
        }
    return out


def main():
    res = []
    for arg in sys.argv[2:]:
        seq, run = arg.split(":", 1)
        r = analyse(int(seq), run)
        res.append(r)
        m = r["riscs"]["math"]
        print(
            f"{r['seq']:>6} chunk {r['chunk']:>3}  SDPA {r['sdpa_kernel_us']:10.1f} us  coverage {m['coverage_pct']:5.1f}%  "
            + "  ".join(f"{s[5:]} {v['share_pct']:5.2f}%" for s, v in m["stages"].items())
        )
        print("        math mean us/exec: " + "  ".join(f"{s[5:]} {v['mean_us']:.2f}" for s, v in m["stages"].items()))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print(f"[json] {sys.argv[1]}")


if __name__ == "__main__":
    main()
