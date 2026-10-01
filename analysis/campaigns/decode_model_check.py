#!/usr/bin/env python3
"""Join the decode walls measured on this tree to the polaris model and to the published campaign.

The decode counterpart of rebaseline_table.py. Offline, no arguments, reads $DD and writes beside the
captures. Two blocks, either of which may be absent:

  t28_*  paged decode (analysis/decode_sweep.py), reduced by reduce_decode.py into decode_sweep_table.csv
  r1b_*  non-paged decode (analysis/r1_decode.py), read straight from its ops report

The published column is parsed out of the DECODE_SWEEP_T28 / DECODE_NONPAGED_R1B tables of the polaris
test suite, which is where the campaign's measured decode walls live; the file is parsed, never imported,
so no polaris venv is needed. The model column calls predict_decode from $POLARIS, as zone_decomp.py does.

The wall of a position is the median of invocations 1 and 2, invocation 0 being a warmup, matching
reduce_decode.py and the convention the campaign reduced with.
"""

import ast
import csv
import os
import re
import sys
from pathlib import Path

_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))
PINS = POLARIS / "tests" / "test_perf" / "test_roofline_sdpa.py"
OUT = DD / "decode_model_check.csv"
CLK = 1350.0

sys.path.insert(0, str(POLARIS / "ttsim" / "perf"))
from roofline_sdpa import predict_decode  # noqa: E402

# The harness geometry, matching analysis/decode_sweep.py: HiFi2, fp32 DEST, auto k chunk, 32-row pages.
T28 = dict(num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=0, fidelity="HiFi2",
           input_dtype="bfloat16", accum_dtype="float32", paged=True, page_block_size=32,
           max_cores_per_head_batch=16)
# analysis/r1_decode.py: flat cache, Q bf16 in DRAM, k chunk 128, HiFi2, cur_pos = cache - 1.
R1B = dict(num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=128, fidelity="HiFi2",
           input_dtype="bfloat16")


def published(name):
    """{key: measured us} for one of the pinned decode tables in the polaris test suite."""
    if not PINS.is_file():
        return {}
    for node in ast.walk(ast.parse(PINS.read_text())):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return {tuple(t[:-1]): t[-1] for t in ast.literal_eval(node.value)}
    return {}


def rows_t28():
    """Measured paged points, from the reduce_decode.py table."""
    src = DD / "decode_sweep_table.csv"
    if not src.is_file():
        return []
    pub = published("DECODE_SWEEP_T28")
    out = []
    for r in csv.DictReader(src.open()):
        gx, gy = (int(v) for v in r["grid"].split("x"))
        cores, batch, kvd, pos = gx * gy, int(r["batch"]), r["kv_dtype"], int(r["position"])
        p = predict_decode(cache_len=16384, cur_pos=pos, batch=batch, num_cores=cores,
                           **{**T28, "kv_input_dtype": kvd})
        out.append(dict(block="T2.8 paged", config=f"g{cores} b{batch} {kvd} pos{pos}",
                        measured_us=float(r["dur_ns_median_excl_first"]) / 1000.0,
                        published_us=pub.get((cores, batch, kvd, pos)),
                        model_us=p.wall_clock_cycles / CLK,
                        kv_GBps_measured=float(r["kv_GBps"]),
                        kv_GBps_model=p.config_echo["kv_stream_gbps"],
                        active_cores=p.config_echo["active_cores"],
                        k_chunk=p.config_echo["k_chunk"]))
    return out


def rows_r1b():
    """Measured non-paged points, straight from the r1b ops reports."""
    import pandas as pd

    pub = published("DECODE_NONPAGED_R1B")
    out = []
    for f in sorted(DD.glob("r1b_decode_nonpaged_*_ops_perf_results.csv")):
        tag = f.name.replace("_ops_perf_results.csv", "")
        prov = (DD / f"{tag}.csv").open().readline()
        env = dict(re.findall(r"DEC_(\w+)=(\S+)", prov))
        batch, kvd = int(env.get("B", 32)), env.get("KV_DTYPE", "bfloat16")
        caches = [int(x) for x in env.get("CACHE", env.get("POS", "1024")).split(",")]
        iters = int(env.get("ITERS", 3))
        o = pd.read_csv(f, low_memory=False)
        dur = next(c for c in o.columns if c.startswith("DEVICE KERNEL DURATION"))
        s = o[o["OP CODE"].astype(str).str.contains("SdpaDecode|ScaledDotProductAttentionDecode",
                                                    regex=True)].reset_index(drop=True)
        for i, cache in enumerate(caches):
            g = s.iloc[i * iters:(i + 1) * iters]
            if len(g) < iters:
                continue
            med = g[dur].astype(float).iloc[1:].median()
            p = predict_decode(cache_len=cache, cur_pos=cache - 1, batch=batch, num_cores=110,
                               **{**R1B, "kv_input_dtype": kvd})
            out.append(dict(block="R1b non-paged", config=f"b{batch} {kvd} cache{cache}",
                            measured_us=med / 1000.0, published_us=pub.get((batch, kvd, cache)),
                            model_us=p.wall_clock_cycles / CLK, kv_GBps_measured=None,
                            kv_GBps_model=p.config_echo["kv_stream_gbps"],
                            active_cores=p.config_echo["active_cores"],
                            k_chunk=p.config_echo["k_chunk"]))
    return out


def pct(a, b):
    return None if not a or not b else (a - b) / b * 100.0


def main():
    rows = rows_t28() + rows_r1b()
    if not rows:
        print(f"no decode captures in {DD}")
        return
    for r in rows:
        r["model_vs_measured_pct"] = pct(r["model_us"], r["measured_us"])
        r["measured_vs_published_pct"] = pct(r["measured_us"], r["published_us"])
    cols = ["block", "config", "published_us", "measured_us", "model_us", "model_vs_measured_pct",
            "measured_vs_published_pct", "kv_GBps_measured", "kv_GBps_model", "active_cores", "k_chunk"]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r.get(k) for k in cols} for r in rows)

    def fmt(v, nd=1):
        return "-" if v is None else f"{v:.{nd}f}"

    width = max(len(r["config"]) for r in rows) + 2
    print(f"{'config':<{width}}{'published':>10}{'measured':>10}{'model':>10}"
          f"{'model/meas':>12}{'meas/pub':>10}")
    for block in dict.fromkeys(r["block"] for r in rows):
        print(f"-- {block}")
        for r in (x for x in rows if x["block"] == block):
            print(f"{r['config']:<{width}}{fmt(r['published_us']):>10}{fmt(r['measured_us']):>10}"
                  f"{fmt(r['model_us']):>10}{fmt(r['model_vs_measured_pct'], 2):>11}%"
                  f"{fmt(r['measured_vs_published_pct'], 2):>9}%")
    err = [abs(r["model_vs_measured_pct"]) for r in rows if r["model_vs_measured_pct"] is not None]
    print(f"\nmodel vs measured: mean abs {sum(err) / len(err):.2f}%, worst {max(err):.2f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
