#!/usr/bin/env python3
"""
FPU / SFPU utilization report built directly from the raw tt-metal profiler logs.

What it answers (per run directory, e.g. generated/profiler/prefill_128_1dev):

  * FPU  utilization of Matmul ops   (FPU_COUNTER  / counter window)
  * SFPU utilization of SDPA   ops   (SFPU_COUNTER / counter window)
  * share of Matmul ops / SDPA ops in the total device kernel time

Utilization is reported two ways, per op, and then aggregated over the ops of a category:

  active-core util     for every core that ran the op:   counter_c / ref_cnt_c     (ref_cnt_c = that core's own counter window,
                       which equals its BRISC kernel window). Median over the cores that ran the op.
  full-grid-core util  for every core of the worker grid: counter_c / T_op         (T_op = op kernel duration = first kernel start
                       to last kernel end over all cores; cores that did not run the op contribute 0).
                       Median over ALL grid cores (and, as a reference, the mean).

Inputs are read from the profiler report directory (nothing is taken from the pre-aggregated per-op utilization columns,
those are only used afterwards as a cross-check):

  reports/<ts>/profile_log_device.csv        raw per-core perf-counter records (timer id 9090) + kernel zones
  reports/<ts>/ops_perf_results_<ts>.csv     op list: order, op code, kernel duration, trace/replay ids, shapes

Usage:
  python tools/tracy/perf_util_report.py generated/profiler/prefill_128_1dev
  python tools/tracy/perf_util_report.py RUN_A RUN_B RUN_C --json out.json      # several runs + comparison table
  python tools/tracy/perf_util_report.py RUN --select last-forward              # include eager tail (final norm + lm_head)

Needs only pandas + numpy.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PERF_COUNTER_TIMER_ID = 9090
COUNTERS = ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER")
SHORT = {"FPU_COUNTER": "FPU", "SFPU_COUNTER": "SFPU", "MATH_COUNTER": "MATH"}


# --------------------------------------------------------------------------------------------------------------------
# Locating / loading inputs
# --------------------------------------------------------------------------------------------------------------------
def find_report(run_dir: Path, report_name=None):
    """Return (report_dir, ops_csv, device_log) for a profiler run directory (or directly a report directory)."""
    run_dir = Path(run_dir)
    if list(run_dir.glob("ops_perf_results_*.csv")):
        report_dir = run_dir
    else:
        reports = run_dir / "reports"
        if not reports.is_dir():
            sys.exit(f"[error] {run_dir}: no ops_perf_results_*.csv and no reports/ directory")
        cands = sorted(d for d in reports.iterdir() if d.is_dir() and list(d.glob("ops_perf_results_*.csv")))
        if report_name:
            cands = [d for d in cands if d.name == report_name]
        if not cands:
            sys.exit(f"[error] {run_dir}: no report directory with an ops_perf_results csv")
        report_dir = cands[-1]  # timestamped names sort chronologically -> latest
    ops_csv = sorted(report_dir.glob("ops_perf_results_*.csv"))[-1]
    dev_log = report_dir / "profile_log_device.csv"
    if not dev_log.exists():
        alt = run_dir / ".logs" / "profile_log_device.csv"
        if alt.exists():
            dev_log = alt
        else:
            sys.exit(f"[error] {report_dir}: profile_log_device.csv not found")
    return report_dir, ops_csv, dev_log


def read_device_header(dev_log: Path):
    """ARCH, CHIP_FREQ[MHz], Max Compute Cores from line 1; column names from line 2."""
    with open(dev_log) as f:
        first = f.readline()
        second = f.readline()
    m = re.search(r"ARCH:\s*(\w+).*CHIP_FREQ\[MHz\]:\s*([\d.]+).*Max Compute Cores:\s*(\d+)", first)
    if not m:
        sys.exit(f"[error] {dev_log}: unexpected first line: {first!r}")
    cols = [c.strip() for c in second.rstrip("\n").split(",")]
    return m.group(1), float(m.group(2)), int(m.group(3)), cols


def load_device_log(dev_log: Path, cols):
    want = [
        "PCIe slot",
        "core_x",
        "core_y",
        "RISC processor type",
        "timer_id",
        "time[cycles since reset]",
        "run host ID",
        "trace id counter",
        "zone name",
        "type",
        "meta data",
    ]
    idx = [cols.index(c) for c in want]
    df = pd.read_csv(
        dev_log,
        skiprows=2,
        header=None,
        names=cols,
        usecols=idx,
        dtype={
            "PCIe slot": "int16",
            "core_x": "int16",
            "core_y": "int16",
            "RISC processor type": "string",
            "timer_id": "int32",
            "time[cycles since reset]": "int64",
            "run host ID": "int64",
            "trace id counter": "float64",
            "zone name": "string",
            "type": "string",
            "meta data": "string",
        },
        low_memory=False,
    )
    df = df.rename(
        columns={
            "PCIe slot": "dev",
            "RISC processor type": "risc",
            "time[cycles since reset]": "ts",
            "run host ID": "call",
            "trace id counter": "sess",
            "zone name": "zone",
        }
    )
    df["sess"] = df["sess"].fillna(-1).astype("int32")
    return df


def counter_name_map():
    """Numeric counter-type ids -> names; only needed for logs that store ids instead of names."""
    try:
        sys.path.append(str(Path(__file__).resolve().parents[2] / "tt_metal" / "tt-llk" / "tools" / "python"))
        from tt_llk_perf.headers import counter_type_names

        return counter_type_names()
    except Exception:
        return {}


def extract_perf_records(dev: pd.DataFrame):
    """Raw perf-counter rows -> long frame (dev, call, sess, x, y, risc, neo, ctr, val, ref)."""
    perf = dev[dev["timer_id"] == PERF_COUNTER_TIMER_ID]
    if perf.empty:
        return None
    md = perf["meta data"]
    ctr = md.str.extract(r'"counter type":"?([^";}]+)"?', expand=False)
    if ctr.str.fullmatch(r"\d+").any():
        names = counter_name_map()
        ctr = ctr.map(lambda v: names.get(int(v), f"UNKNOWN_{v}") if isinstance(v, str) and v.isdigit() else v)
    neo = md.str.extract(r'"neo":(\d+)', expand=False).fillna("0").astype(int)
    ref = md.str.extract(r'"ref cnt":(\d+)', expand=False).astype("int64")
    val = md.str.extract(r'"value":(\d+)', expand=False).astype("int64")
    out = pd.DataFrame(
        {
            "dev": perf["dev"].to_numpy(),
            "call": perf["call"].to_numpy(),
            "sess": perf["sess"].to_numpy(),
            "x": perf["core_x"].to_numpy(),
            "y": perf["core_y"].to_numpy(),
            "risc": perf["risc"].astype(str).to_numpy(),
            "neo": neo.to_numpy(),
            "ctr": ctr.astype(str).to_numpy(),
            "val": val.to_numpy(),
            "ref": ref.to_numpy(),
        }
    )
    return out


def extract_kernel_windows(dev: pd.DataFrame):
    """Per (dev, call, sess): op kernel window [first kernel start, last kernel end] in cycles over all cores/RISCs.
    Also the set of worker (Tensix) cores seen per device."""
    k = dev[dev["zone"].astype("string").str.endswith("-KERNEL", na=False)]
    win = (
        k.groupby(["dev", "call", "sess"])["ts"].agg(t_start="min", t_end="max").assign(T=lambda d: d.t_end - d.t_start)
    )
    tensix = k[k["risc"].astype(str).str.match(r"^(BRISC|NCRISC|TRISC)")]
    grid = tensix.groupby("dev").apply(lambda d: len(d[["core_x", "core_y"]].drop_duplicates()), include_groups=False)
    return win, grid.to_dict()


# --------------------------------------------------------------------------------------------------------------------
# Op table
# --------------------------------------------------------------------------------------------------------------------
def pad_dim(v):
    """'128[128]' -> 128"""
    if pd.isna(v):
        return None
    m = re.match(r"\s*(\d+)", str(v))
    return int(m.group(1)) if m else None


def shape_label(row, cat):
    def g(n, d):
        return pad_dim(row.get(f"INPUT_{n}_{d}_PAD[LOGICAL]"))

    if cat == "matmul":
        m, k, n = g(0, "Y"), g(0, "X"), g(1, "X")
        if None not in (m, k, n):
            batch = g(0, "Z") or 1
            return f"{m}x{k}x{n}" + (f" (b{batch})" if batch > 1 else "")
    dims = [g(0, d) for d in "WZYX"]
    if all(v is not None for v in dims):
        return "x".join(str(v) for v in dims)
    return ""


def classify(code, matmul_re, sdpa_re):
    if sdpa_re.search(code):
        return "sdpa"
    if matmul_re.search(code):
        return "matmul"
    return "other"


def load_ops(ops_csv: Path, matmul_re, sdpa_re):
    ops = pd.read_csv(ops_csv, low_memory=False)
    ops["idx"] = np.arange(len(ops))
    ops["dev"] = ops["DEVICE ID"].astype(int)
    ops["call"] = ops["GLOBAL CALL COUNT"].astype("int64")
    ops["sess"] = ops["METAL TRACE REPLAY SESSION ID"].fillna(-1).astype(int)
    ops["trace_id"] = ops["METAL TRACE ID"]
    ops["kern_ns"] = ops["DEVICE KERNEL DURATION [ns]"].astype(float)
    ops["code"] = ops["OP CODE"].str.replace(r"(DeviceOperation|Operation)$", "", regex=True)
    ops["kind"] = [classify(c, matmul_re, sdpa_re) for c in ops["OP CODE"]]
    ops["shape"] = [shape_label(r, r["kind"]) for r in ops.to_dict("records")]
    return ops


def select_scope(ops_dev: pd.DataFrame, mode: str, session, idx_range, fwd_start_re):
    """Pick the ops that make up 'the run'. Returns (selected, mode_used, description, eager_tail)."""
    if idx_range:
        lo, hi = idx_range
        sel = ops_dev[(ops_dev.idx >= lo) & (ops_dev.idx <= hi)]
        return sel, "range", f"op rows {lo}..{hi} of the ops csv", ops_dev.iloc[0:0]

    sessions = sorted(s for s in ops_dev.sess.unique() if s >= 0)
    if mode == "auto":
        mode = "trace" if sessions else "last-forward"

    if mode == "all":
        return ops_dev, mode, "all ops in the log (includes compile/warmup passes!)", ops_dev.iloc[0:0]

    if mode == "trace":
        if not sessions:
            sys.exit("[error] --select trace requested but the log has no trace replays")
        s = session if session is not None else sessions[-1]
        sel = ops_dev[ops_dev.sess == s]
        tids = sorted(sel.trace_id.dropna().unique().astype(int).tolist())
        desc = f"metal-trace replay session {s} (trace id {tids}); {len(sessions)} replay session(s) in log"
        tail = ops_dev[(ops_dev.idx > sel.idx.max()) & (ops_dev.sess < 0)]
        return sel, mode, desc, tail

    if mode == "last-forward":
        starts = ops_dev[ops_dev.code.str.contains(fwd_start_re)]
        if starts.empty:
            print(f"[warn] no op matching /{fwd_start_re.pattern}/ found; using all ops", file=sys.stderr)
            return ops_dev, "all", "all ops (no forward-start op found)", ops_dev.iloc[0:0]
        i0 = starts.idx.max()
        sel = ops_dev[ops_dev.idx >= i0]
        desc = f"last forward pass: from the last '{fwd_start_re.pattern}' op (row {i0}) to the end of the log"
        return sel, mode, desc, ops_dev.iloc[0:0]

    sys.exit(f"[error] unknown --select {mode}")


# --------------------------------------------------------------------------------------------------------------------
# Per-op utilization statistics
# --------------------------------------------------------------------------------------------------------------------
def per_op_stats(wide, key, T_cycles, grid_n, header_cores):
    """Stats for one op from its per-core counter rows (index = cores). Returns dict (NaN-free keys only when data)."""
    g = wide.loc[key]
    out = {"n_active": int(len(g))}
    for c in COUNTERS:
        if ("val", c) not in g.columns:
            continue
        s = SHORT[c]
        val = g[("val", c)].to_numpy(float)
        ref = g[("ref", c)].to_numpy(float)
        ok = ref > 0
        act = np.full(len(val), np.nan)
        act[ok] = val[ok] / ref[ok] * 100.0
        out[f"{s}_act_med"] = float(np.nanmedian(act)) if ok.any() else np.nan
        out[f"{s}_act_mean"] = float(np.nanmean(act)) if ok.any() else np.nan
        work = ok & (val > 0)
        out[f"{s}_nz"] = int((val > 0).sum())
        out[f"{s}_work_med"] = float(np.median(act[work])) if work.any() else np.nan
        if T_cycles and T_cycles > 0:
            full = val / T_cycles * 100.0
            n = max(grid_n, len(full))
            full_all = np.concatenate([full, np.zeros(n - len(full))])
            out[f"{s}_full_med"] = float(np.median(full_all))
            out[f"{s}_full_mean"] = float(full_all.mean())
            out[f"{s}_full_max"] = float(full.max()) if len(full) else 0.0
            out[f"{s}_full_mean_hdr"] = float(val.sum() / header_cores / T_cycles * 100.0)  # for CSV cross-check
        else:
            for k in ("full_med", "full_mean", "full_max", "full_mean_hdr"):
                out[f"{s}_{k}"] = np.nan
    return out


def wavg(vals, w):
    vals, w = np.asarray(vals, float), np.asarray(w, float)
    ok = ~np.isnan(vals) & ~np.isnan(w)
    return float(np.average(vals[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else np.nan


def category_agg(df, col):
    """Aggregate a per-op metric over the ops of a category (ops with counter data only)."""
    d = df[df[col].notna()]
    if d.empty:
        return None
    return {
        "op_median": float(d[col].median()),
        "tw_mean": wavg(d[col], d["kern_ns"]),
        "min": float(d[col].min()),
        "max": float(d[col].max()),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------------------------------------
def f2(x, w=7):
    return f"{'n/a':>{w}}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{w}.2f}"


def hr(ch="-", n=128):
    return ch * n


def print_scope_breakdown(sel):
    tot = sel.kern_ns.sum()
    g = (
        sel.groupby("code")
        .agg(n=("idx", "size"), us=("kern_ns", lambda s: s.sum() / 1e3), kind=("kind", "first"))
        .sort_values("us", ascending=False)
    )
    print("\nKernel time by op type (selected scope)")
    print(hr())
    print(f"{'op':<40}{'n':>5}{'kernel us':>13}{'% of total':>12}   category")
    for code, r in g.head(25).iterrows():
        print(f"{code:<40}{int(r.n):>5}{r.us:>13.2f}{r.us * 1e3 / tot * 100:>11.2f}%   {r.kind}")
    if len(g) > 25:
        rest = g.iloc[25:]
        print(
            f"{'... ' + str(len(rest)) + ' more op types':<40}{int(rest.n.sum()):>5}{rest.us.sum():>13.2f}{rest.us.sum() * 1e3 / tot * 100:>11.2f}%"
        )
    print(hr())


def print_per_op_table(st, max_rows):
    d = st[st.kind != "other"]
    if d.empty:
        return
    print("\nPer-op detail: Matmul / SDPA ops in execution order   (all utilizations in %)")
    print("  act-med  = median over the cores that ran the op (counter / own counter window)")
    print("  full-med = median over ALL grid cores, idle cores = 0 (counter / op kernel duration)")
    print("  full-avg = mean   over ALL grid cores, idle cores = 0")
    print(hr(ch="="))
    print(
        f"{'row':>4} {'cat':<6} {'op':<20} {'shape':<22} {'fid':<6}{'cores':>6}{'kern us':>10}{'%tot':>7} |"
        f"{'FPU act-med':>12}{'full-med':>9}{'full-avg':>9} |{'SFPU act-med':>13}{'full-med':>9}{'full-avg':>9}  note"
    )
    print(hr())
    shown = d if len(d) <= max_rows else d.head(max_rows)
    for _, r in shown.iterrows():
        fid = "" if pd.isna(r.get("MATH FIDELITY")) else str(r["MATH FIDELITY"])
        note = ""
        main = "FPU" if r.kind == "matmul" else "SFPU"
        nz, n = r.get(f"{main}_nz"), r.get("n_active")
        if pd.isna(r.get(f"{main}_act_med")):
            note = f"no {main} counter data"
        elif n and nz is not None and nz < 0.9 * n:
            note = (
                f"only {int(nz)}/{int(n)} cores do work ({main}>0); their median = " f"{r.get(main + '_work_med'):.2f}%"
            )
        print(
            f"{int(r.idx):>4} {r.kind:<6} {r.code[:20]:<20} {str(r['shape'])[:22]:<22} {fid:<6}"
            f"{(str(int(r.get('n_active'))) if not pd.isna(r.get('n_active')) else 'n/a'):>6}{r.kern_ns / 1e3:>10.2f}{r.pct_total:>7.2f} |"
            f"{f2(r.get('FPU_act_med'), 12)}{f2(r.get('FPU_full_med'), 9)}{f2(r.get('FPU_full_mean'), 9)} |"
            f"{f2(r.get('SFPU_act_med'), 13)}{f2(r.get('SFPU_full_med'), 9)}{f2(r.get('SFPU_full_mean'), 9)}  {note}"
        )
    if len(d) > max_rows:
        print(f"... {len(d) - max_rows} more rows not shown (use --max-rows or --csv)")
    print(hr(ch="="))


def validate_against_csv(st):
    """Cross-check raw-log numbers against the per-op columns the profiler already put in the ops csv."""
    res = {}
    pairs = [
        ("FPU median (act)", "FPU_act_med", "FPU Util Median (%)"),
        ("SFPU median (act)", "SFPU_act_med", "SFPU Util Median (%)"),
        ("FPU full-grid avg (/hdr cores)", "FPU_full_mean_hdr", "Avg FPU util on full grid (%)"),
        ("SFPU full-grid avg (/hdr cores)", "SFPU_full_mean_hdr", "Avg SFPU util on full grid (%)"),
    ]
    for label, mine, theirs in pairs:
        if theirs not in st.columns or mine not in st.columns:
            continue
        d = st[st[mine].notna() & st[theirs].notna()]
        if d.empty:
            continue
        diff = (d[mine] - d[theirs]).abs()
        res[label] = (float(diff.max()), int(len(d)))
    return res


def make_report(run_dir, args, matmul_re, sdpa_re, fwd_re):
    report_dir, ops_csv, dev_log = find_report(run_dir, args.report)
    arch, freq_mhz, hdr_cores, cols = read_device_header(dev_log)
    print(f"\n[load] {dev_log}  ({dev_log.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
    dev = load_device_log(dev_log, cols)
    perf = extract_perf_records(dev)
    if perf is None:
        sys.exit(
            f"[error] {run_dir}: no perf-counter records (timer id {PERF_COUNTER_TIMER_ID}) in the device log.\n"
            "        Re-run with the profiler perf-counter capture enabled (--profiler-capture-perf-counters=fpu)."
        )
    win, grid_seen = extract_kernel_windows(dev)
    del dev
    ops = load_ops(ops_csv, matmul_re, sdpa_re)

    # one reader per (core, risc, neo); wide frame: index (dev, call, sess, x, y, risc, neo), columns (val|ref, counter)
    wide = perf.pivot_table(
        index=["dev", "call", "sess", "x", "y", "risc", "neo"], columns="ctr", values=["val", "ref"], aggfunc="first"
    )
    present = [c for c in COUNTERS if ("val", c) in wide.columns]
    wide = wide.sort_index()
    have = set(wide.droplevel(["x", "y", "risc", "neo"]).index.unique())

    results = []
    for dev_id in sorted(ops.dev.unique()):
        ops_dev = ops[ops.dev == dev_id].copy()
        grid_n = args.grid_cores or grid_seen.get(dev_id) or hdr_cores
        grid_src = "--grid-cores" if args.grid_cores else f"distinct worker cores seen in the device log"
        sel, mode, desc, tail = select_scope(ops_dev, args.select, args.session, args.ops_range, fwd_re)
        if sel.empty:
            print(f"[warn] device {dev_id}: empty selection", file=sys.stderr)
            continue

        # per-op stats
        rows = []
        for r in sel.itertuples():
            key = (r.dev, r.call, r.sess)
            T = win["T"].get((r.dev, r.call, r.sess), np.nan) if (r.dev, r.call, r.sess) in win.index else np.nan
            if np.isnan(T):
                T = r.kern_ns * freq_mhz / 1e3 if not np.isnan(r.kern_ns) else np.nan
            st = {"T_cycles": T}
            if key in have:
                st.update(per_op_stats(wide, key, T, grid_n, hdr_cores))
            rows.append(st)
        st = pd.concat([sel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        total_ns = st.kern_ns.sum()
        st["pct_total"] = st.kern_ns / total_ns * 100.0

        # sanity: raw-log kernel window vs csv kernel duration
        kd = (st["T_cycles"] / freq_mhz * 1e3 - st["kern_ns"]).abs() / st["kern_ns"].replace(0, np.nan)
        kd_max = float(kd.max()) if kd.notna().any() else np.nan
        val = validate_against_csv(st)

        # categories
        cats = {}
        for cat in ("matmul", "sdpa"):
            d = st[st.kind == cat]
            c = {
                "n_ops": int(len(d)),
                "n_with_data": int(d["n_active"].notna().sum()) if "n_active" in d else 0,
                "kern_ns": float(d.kern_ns.sum()),
                "pct_total": float(d.kern_ns.sum() / total_ns * 100.0) if total_ns else np.nan,
                "by_code": {k: int(v) for k, v in d.code.value_counts().items()},
            }
            for ctr in ("FPU", "SFPU", "MATH"):
                for m in ("act_med", "work_med", "act_mean", "full_med", "full_mean"):
                    col = f"{ctr}_{m}"
                    c[col] = category_agg(d, col) if col in d.columns else None
            cats[cat] = c

        tail_info = None
        if len(tail):
            tn = tail.kern_ns.sum()
            t_mm = tail[tail.kind == "matmul"].kern_ns.sum()
            t_sd = tail[tail.kind == "sdpa"].kern_ns.sum()
            tot2 = total_ns + tn
            tail_info = {
                "n_ops": int(len(tail)),
                "kern_ns": float(tn),
                "pct_matmul_if_included": float((cats["matmul"]["kern_ns"] + t_mm) / tot2 * 100),
                "pct_sdpa_if_included": float((cats["sdpa"]["kern_ns"] + t_sd) / tot2 * 100),
                "tail_matmul_ops": int((tail.kind == "matmul").sum()),
            }

        results.append(
            dict(
                run=str(run_dir),
                run_name=Path(run_dir).name,
                report=report_dir.name,
                device=int(dev_id),
                arch=arch,
                freq_mhz=freq_mhz,
                header_cores=hdr_cores,
                grid_cores=int(grid_n),
                grid_src=grid_src,
                select=mode,
                scope_desc=desc,
                n_ops=int(len(st)),
                n_ops_with_counters=int(st["n_active"].notna().sum()) if "n_active" in st else 0,
                total_kernel_us=float(total_ns / 1e3),
                cats=cats,
                tail=tail_info,
                validation=val,
                kernel_window_max_rel_diff=kd_max,
                _st=st,
            )
        )
    return results


def print_report(res, args):
    st = res["_st"]
    print("\n" + hr("="))
    print(f"FPU / SFPU UTILIZATION REPORT  -  {res['run_name']}   (device {res['device']}, report {res['report']})")
    print(hr("="))
    print(
        f"arch {res['arch']}, {res['freq_mhz']:.0f} MHz | full grid = {res['grid_cores']} worker cores "
        f"({res['grid_src']}; profiler header says {res['header_cores']}, which also counts dispatch cores)"
    )
    print(f"scope : {res['scope_desc']}")
    print(
        f"        {res['n_ops']} ops ({res['n_ops_with_counters']} with perf-counter data), "
        f"total device kernel time {res['total_kernel_us']:.2f} us"
    )

    print_scope_breakdown(st)
    print_per_op_table(st, args.max_rows)

    if res["validation"] or not np.isnan(res["kernel_window_max_rel_diff"]):
        print("\nCross-check of this raw-log analysis against the profiler's own ops csv columns (selected ops)")
        print(
            f"  kernel window from raw zones vs 'DEVICE KERNEL DURATION [ns]' : max rel diff {res['kernel_window_max_rel_diff']:.2e}"
        )
        for k, (mx, n) in res["validation"].items():
            print(f"  {k:<34}: max |diff| = {mx:.3g} percentage points over {n} ops")

    # ------------------------------------------------------------------ summary
    mm, sd = res["cats"]["matmul"], res["cats"]["sdpa"]
    W = 100
    print("\n" + hr("=", W))
    print(f"SUMMARY  -  {res['run_name']}".center(W))
    print(hr("=", W))
    print(f"scope: {res['scope_desc']}")
    print(f"total device kernel time of scope: {res['total_kernel_us']:.2f} us over {res['n_ops']} ops")
    print(hr("-", W))
    print(f"{'share of total kernel time':<34}{'ops':>5}{'kernel us':>14}{'% of total':>13}")
    for name, c in (("Matmul (all matmul variants)", mm), ("SDPA", sd)):
        codes = ", ".join(f"{k} x{v}" for k, v in c["by_code"].items()) or "-"
        print(f"{name:<34}{c['n_ops']:>5}{c['kern_ns'] / 1e3:>14.2f}{c['pct_total']:>12.2f}%   [{codes}]")
    oth_ns = res["total_kernel_us"] * 1e3 - mm["kern_ns"] - sd["kern_ns"]
    print(
        f"{'everything else':<34}{res['n_ops'] - mm['n_ops'] - sd['n_ops']:>5}{oth_ns / 1e3:>14.2f}"
        f"{oth_ns / (res['total_kernel_us'] * 1e3) * 100:>12.2f}%"
    )

    def block(title, c, ctr):
        print(hr("-", W))
        print(f"{title}   ({c['n_with_data']}/{c['n_ops']} ops have counter data)")
        print(f"{'':<54}{'tw-avg':>9}{'med':>9}{'min':>9}{'max':>9}   <- over the ops")
        for label, m in (
            ("median per ACTIVE core (cores that ran the op)", "act_med"),
            ("median per WORKING core (cores with counter > 0)", "work_med"),
            ("mean   per ACTIVE core", "act_mean"),
            ("median per FULL-GRID core (idle cores = 0)", "full_med"),
            ("mean   per FULL-GRID core (idle cores = 0)", "full_mean"),
        ):
            a = c.get(f"{ctr}_{m}")
            if a is None:
                print(f"{label:<54}{'n/a':>9}")
            else:
                print(f"{label:<54}{a['tw_mean']:>8.2f}%{a['op_median']:>8.2f}%{a['min']:>8.2f}%{a['max']:>8.2f}%")

    block("FPU utilization - MATMUL ops", mm, "FPU")
    block("SFPU utilization - SDPA ops", sd, "SFPU")
    print(hr("-", W))
    print("supplementary (same definitions):")
    for name, c, ctr in (("SDPA   FPU ", sd, "FPU"), ("Matmul SFPU", mm, "SFPU")):
        a, f = c.get(f"{ctr}_act_med"), c.get(f"{ctr}_full_mean")
        if a and f:
            w = c.get(f"{ctr}_work_med")
            wtxt = f"{w['tw_mean']:.2f}%" if w else "n/a (no core has counter > 0)"
            print(
                f"  {name}: median per active core {a['tw_mean']:.2f}% / per working core {wtxt} (time-weighted), "
                f"full-grid mean {f['tw_mean']:.2f}%"
            )
    if res["tail"]:
        t = res["tail"]
        print(hr("-", W))
        print(
            f"not in scope: {t['n_ops']} eager ops after the replay (final norm / lm_head ...), "
            f"{t['kern_ns'] / 1e3:.2f} us, incl. {t['tail_matmul_ops']} matmuls."
        )
        print(
            f"  if they were included: Matmul {t['pct_matmul_if_included']:.2f}% / SDPA {t['pct_sdpa_if_included']:.2f}% "
            f"of kernel time  (re-run with --select last-forward for full stats)"
        )
    print(hr("-", W))
    print("notes:")
    print(" * 'active' = cores that launched the op and reported counters; 'full grid' adds the idle cores of the")
    print(
        f"   {res['grid_cores']}-core worker grid as 0%. The full-grid MEDIAN is 0 whenever fewer than half of the grid ran the op;"
    )
    print("   the full-grid MEAN is the aggregate chip utilization and is the more informative number in that case.")
    print(
        " * per op the statistic is taken over cores; across the ops of a category it is aggregated as tw-avg (average weighted by"
    )
    print("   op kernel time), med (plain median over ops), min / max (over ops).")
    print(
        " * WORKING = ACTIVE cores whose counter is non-zero. They differ when an op launches cores that get no work (e.g. SDPA at"
    )
    print(
        "   short sequence length); the ACTIVE median is then a mid-point between two clusters, WORKING shows the busy cores."
    )
    print(hr("=", W))


def print_comparison(all_res):
    print("\n" + hr("=", 150))
    print(
        "COMPARISON ACROSS RUNS (time-weighted over ops; act = median per active core, full = median / mean per full-grid core)"
    )
    print(hr("=", 150))
    print(
        f"{'run':<28}{'dev':>4}{'scope':>14}{'ops':>5}{'kern us':>11} |{'MM %t':>7}{'FPU act':>9}{'FPU full-med':>13}{'FPU full-avg':>13}"
        f" |{'SDPA %t':>8}{'SFPU act':>9}{'SFPU full-med':>14}{'SFPU full-avg':>14}"
    )
    print(hr("-", 150))

    def g(c, key):
        a = c.get(key)
        return f2(a["tw_mean"], 0) if a else "n/a"

    for r in all_res:
        mm, sd = r["cats"]["matmul"], r["cats"]["sdpa"]
        print(
            f"{r['run_name'][:28]:<28}{r['device']:>4}{r['select']:>14}{r['n_ops']:>5}{r['total_kernel_us']:>11.1f} |"
            f"{mm['pct_total']:>7.2f}{g(mm, 'FPU_act_med'):>9}{g(mm, 'FPU_full_med'):>13}{g(mm, 'FPU_full_mean'):>13}"
            f" |{sd['pct_total']:>8.2f}{g(sd, 'SFPU_act_med'):>9}{g(sd, 'SFPU_full_med'):>14}{g(sd, 'SFPU_full_mean'):>14}"
        )
    print(hr("=", 150))


# --------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="profiler run dir(s) (containing reports/) or report dir(s)")
    ap.add_argument("--report", help="report sub-directory name under reports/ (default: latest)")
    ap.add_argument(
        "--select",
        default="auto",
        choices=["auto", "trace", "last-forward", "all"],
        help="which ops form 'the run'. auto = last metal-trace replay if present, else last-forward. "
        "trace = ops of one trace replay session; last-forward = from the last Embeddings op to the end "
        "(includes the eager tail after a replay); all = everything in the log (incl. compile/warmup).",
    )
    ap.add_argument("--session", type=int, help="trace replay session id for --select trace (default: last)")
    ap.add_argument("--ops-range", help="explicit op rows START:END (inclusive, ops csv order) - overrides --select")
    ap.add_argument("--forward-start-op", default="Embeddings", help="regex of the op that starts a forward pass")
    ap.add_argument(
        "--matmul-pattern", default="matmul", help="regex (case-insens.) on OP CODE for the Matmul category"
    )
    ap.add_argument("--sdpa-pattern", default="sdpa|scaleddotproduct", help="regex (case-insens.) on OP CODE for SDPA")
    ap.add_argument(
        "--grid-cores", type=int, help="size of the full worker grid (default: distinct cores seen in the log)"
    )
    ap.add_argument("--max-rows", type=int, default=80, help="max rows in the per-op table")
    ap.add_argument("--json", help="write machine-readable summary (all runs) to this file")
    ap.add_argument(
        "--csv", help="write the per-op table (selected ops) to this csv; with several runs a suffix is added"
    )
    args = ap.parse_args()

    matmul_re = re.compile(args.matmul_pattern, re.I)
    sdpa_re = re.compile(args.sdpa_pattern, re.I)
    fwd_re = re.compile(args.forward_start_op)
    args.ops_range = tuple(int(x) for x in args.ops_range.split(":")) if args.ops_range else None

    all_res = []
    for run in args.runs:
        for res in make_report(run, args, matmul_re, sdpa_re, fwd_re):
            print_report(res, args)
            if args.csv:
                p = Path(args.csv)
                if len(args.runs) > 1 or res["device"]:
                    p = p.with_name(f"{p.stem}_{res['run_name']}_dev{res['device']}{p.suffix}")
                res["_st"].drop(columns=["ATTRIBUTES"], errors="ignore").to_csv(p, index=False)
                print(f"[csv] per-op table -> {p}")
            all_res.append(res)

    if len(all_res) > 1:
        print_comparison(all_res)

    if args.json:
        clean = [{k: v for k, v in r.items() if k != "_st"} for r in all_res]
        with open(args.json, "w") as f:
            json.dump(clean, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
        print(f"[json] summary -> {args.json}")


if __name__ == "__main__":
    main()
