# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correlate RISC-V baby-core HPM counters against Tensix engine counters, across configs.

Input is a campaign_pmc_full.sh capture in $DD: per config, one <cfg>__<mp_pass>_counters.csv
holding all 155 Tensix counters, and one <cfg>__<pass>_riscv_pmc.csv per RISC-V event pair.

Method, and why it is this one
------------------------------
The obvious analysis -- correlate a RISC-V stall fraction against a Tensix utilisation across the
110 cores of a single config -- does not work and was tried. Two per-cycle fractions sharing a
denominator correlate at |r| ~ 1 whatever the mechanism, and on this workload the per-core cycle
count varies 2.09x purely because causal masking gives each core a different number of q chunks.
Partialling the cycle count out does not help, because the relationship is an accounting identity
rather than confounding. See BRISC_PERF_COUNTERS.md, "Do not correlate per-cycle fractions across
cores".

So the unit of observation here is the CONFIG, not the core. Each config contributes one point:
the cross-core mean of each metric. Varying a config knob genuinely changes the physics, so a
correlation across configs can carry a mechanism. Alongside the per-cycle fractions we carry
work-normalised forms (per Tensix instruction issued, per q chunk), because those are what
separated a fixed per-instruction cost from real contention last time.

Ablation configs are held out of the correlation and reported as deltas against their baseline:
they are a different kernel, and their value is direct attribution rather than another data point.

Outputs (in $DD):
  pmc_tensix_configs.csv   one row per config: every Tensix metric and every PMC metric
  pmc_tensix_long.csv      one row per config/risc/event: raw, fraction, work-normalised
  pmc_tensix_corr.csv      cross-config Pearson r for every (risc, pmc metric) x tensix metric
  pmc_tensix_ablation.csv  ablation configs as percent deltas against their baseline
  pmc_tensix_report.md     the readable summary

Usage: python pmc_tensix_corr.py [--dd DIR] [--min-configs N]
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Tensix metrics. name -> (expression over counter names, denominator)
# Denominator "ref" is ref_cnt, the counter bank's own reference cycle count, i.e. elapsed cycles
# on that core. It is the only honest denominator for a Tensix counter: mcycle belongs to a
# specific baby core and the Tensix datapath is shared between them.
#
# Naming traps carried over from the RTL audit (CONTEXT.md, 2026-09-17 evening):
#   *_INSTRN_AVAILABLE_* is instruction-buffer backlog, NOT engine utilisation. FPU_INSTRN_
#     AVAILABLE_1 reads 80% at the anchor where FPU_COUNTER reads 34.9%. Kept, prefixed backlog_.
#   WAITING_FOR_* is the pending wait-mask of a sync instruction, not live engine busy.
#   DATA_HAZARD_STALLS_MOVD2A counts NOT-stalled cycles despite its name. Excluded.
#   AVAILABLE_MATH lives in the PACK bank and ANY_THREAD_STALL is the grant of
#     WAITING_FOR_SRCA_CLEAR -- spare grant inputs reused for unrelated signals.
TENSIX_METRICS = {
    # --- engine utilisation: the four the campaign is aimed at, plus context ---
    "fpu_util": "FPU_COUNTER",
    "sfpu_util": "SFPU_COUNTER",
    "math_util": "MATH_COUNTER",
    "packer_busy": "PACKER_BUSY",
    "unpack0_busy": "UNPACK0_BUSY_THREAD0",
    "unpack1_busy": "UNPACK1_BUSY_THREAD0",
    "unpack_busy_sum": "UNPACK0_BUSY_THREAD0 + UNPACK1_BUSY_THREAD0",
    # --- stalls, per Tensix thread. Thread 0 unpack, 1 math, 2 pack. ---
    "stall_thread0_unpack": "THREAD_STALLS_0",
    "stall_thread1_math": "THREAD_STALLS_1",
    "stall_thread2_pack": "THREAD_STALLS_2",
    "stall_any_thread": "ANY_THREAD_STALL",
    # --- who is blocked waiting on whom. These are the "PACKER / UNPACKER stall" quantities. ---
    "wait_for_pack_idle": "WAITING_FOR_PACK_IDLE_0 + WAITING_FOR_PACK_IDLE_1 + WAITING_FOR_PACK_IDLE_2",
    "wait_for_unpack_idle": "WAITING_FOR_UNPACK_IDLE_0 + WAITING_FOR_UNPACK_IDLE_1 + WAITING_FOR_UNPACK_IDLE_2",
    "wait_for_math_idle": "WAITING_FOR_MATH_IDLE_0 + WAITING_FOR_MATH_IDLE_1 + WAITING_FOR_MATH_IDLE_2",
    "wait_for_sfpu_idle": "WAITING_FOR_SFPU_IDLE_0 + WAITING_FOR_SFPU_IDLE_1 + WAITING_FOR_SFPU_IDLE_2",
    "wait_for_thcon_idle": "WAITING_FOR_THCON_IDLE_0 + WAITING_FOR_THCON_IDLE_1 + WAITING_FOR_THCON_IDLE_2",
    # Src register handshake: math waiting for the unpacker to fill Src, and the unpacker waiting
    # for math to release it. This is the unpack/math pipeline interlock seen from both ends.
    "wait_srca_valid": "WAITING_FOR_SRCA_VALID",
    "wait_srcb_valid": "WAITING_FOR_SRCB_VALID",
    "wait_srca_clear": "WAITING_FOR_SRCA_CLEAR",
    "wait_srcb_clear": "WAITING_FOR_SRCB_CLEAR",
    # Circular-buffer semaphore waits: producer blocked on full, consumer blocked on empty.
    "wait_sem_nonzero": "WAITING_FOR_NONZERO_SEM_0 + WAITING_FOR_NONZERO_SEM_1 + WAITING_FOR_NONZERO_SEM_2",
    "wait_sem_nonfull": "WAITING_FOR_NONFULL_SEM_0 + WAITING_FOR_NONFULL_SEM_1 + WAITING_FOR_NONFULL_SEM_2",
    # --- packer datapath, the derived-hardware quantities from pack_lane_refit.py ---
    "dest_read_granted": "DEST_READ_GRANTED_0",
    "dest_read_available": "PACKER_DEST_READ_AVAILABLE",
    # --- instruction-buffer backlog. NOT utilisation; see the note above. ---
    "backlog_fpu": "FPU_INSTRN_AVAILABLE_1",
    "backlog_pack": "PACK_INSTRN_AVAILABLE_2",
    "backlog_unpack": "UNPACK_INSTRN_AVAILABLE_0",
    # --- Src write ports: issued vs accepted, i.e. unpacker output-side pressure ---
    "srca_write_actual": "SRCA_WRITE_ACTUAL",
    "srcb_write_actual": "SRCB_WRITE_ACTUAL",
    "math_instrn_started": "MATH_INSTRN_STARTED",
    "math_src_data_ready": "MATH_SRC_DATA_READY",
}

# Counter used to normalise Tensix work. Math instructions issued is proportional to algorithmic
# work and independent of elapsed time, so dividing by it yields a cost rather than a duty cycle.
TENSIX_WORK = "MATH_INSTRN_STARTED"

# The engine metrics the campaign is aimed at, as cycle shares. Descriptive and what was asked
# for, but mutually collinear -- see the collinearity section of the report.
HEADLINE_TENSIX = [
    "fpu_util",
    "sfpu_util",
    "packer_busy",
    "wait_for_pack_idle",
    "unpack_busy_sum",
    "wait_for_unpack_idle",
]
# The same quantities per unit of Tensix work, plus the target variable. These are the ones a
# correlation can actually speak about, because they no longer share a denominator with it.
HEADLINE_WORK = [
    "cycles_per_work",
    "fpu_util_per_work",
    "sfpu_util_per_work",
    "packer_busy_per_work",
    "wait_for_pack_idle_per_work",
    "unpack_busy_sum_per_work",
    "wait_for_unpack_idle_per_work",
]

# Sixteen of the 70 L1 counters sit on the L1-to-NoC boundary and see every byte crossing into or
# out of the core, DRAM included. bytes = grants x 16, summed over the four channels of a
# direction. The direction names are from the NIU's side: ring0 "outgoing" is data landing in L1.
NOC_READ_INTO_L1 = [
    "L1_0_NOC_RING0_OUTGOING_0_GRANT",
    "L1_0_NOC_RING0_OUTGOING_1_GRANT",
    "L1_2_NOC_RING0_OUTGOING_2_GRANT",
    "L1_2_NOC_RING0_OUTGOING_3_GRANT",
]
NOC_WRITE_OUT_OF_L1 = [
    "L1_1_NOC_RING1_INCOMING_0_GRANT",
    "L1_1_NOC_RING1_INCOMING_1_GRANT",
    "L1_3_NOC_RING1_INCOMING_2_GRANT",
    "L1_3_NOC_RING1_INCOMING_3_GRANT",
]

DOMAIN_SHIFTS = [("BP", 0), ("IF", 4), ("DE", 8), ("DE1", 12), ("EX", 16), ("LS", 20), ("TTCSR", 24), ("MEM", 28)]
EVENTS = {
    "BP": ["none", "pc_stall_any", "bp_stall_no_id_rtr", "bp_stall_no_mem_rtr", "bp_stall_pc_fifo_full",
           "bp_stall_bp_fifo_full", "btb_hit", "pred_jalr", "pred_jal", "pred_br"],
    "IF": ["none", "if_stall_any", "if_stall_no_id_rtr", "if_stall_lq_full", "if_stall_lq_sb_full",
           "req_stall_any", "req_stall_no_mem_rtr"],
    "DE": ["none", "de_stall_any", "de_stall_no_unit_rtr", "de_stall_if_empty", "de_stall_sync",
           "de_stall_hazard", "hazard_int", "hazard_fp", "hazard_vec", "reg_data_forward",
           "mem_disp_ld", "mem_disp_st", "mem_disp_vec_ld", "mem_disp_vec_st", "ex_disp_mul", "ex_disp_div"],
    "DE1": ["none", "ex_disp_csr", "ex_disp_i", "ex_disp_m", "ex_disp_a", "ex_disp_zb", "fp_disp_s",
            "fp_disp_zfh", "vec_disp_vset", "vec_disp_int", "vec_disp_fixed", "vec_disp_float",
            "vec_disp_ldst", "tti_issue", "vec_replay", "illegal_op"],
    "EX": ["none", "ex_stall_any", "ex_stall_no_instr", "mispred_any", "mispred_new_br", "mispred_dir",
           "mispred_tgt_cond", "mispred_tgt_uncond", "inst_ret"],
    "LS": ["none", "ls_stall_any", "ls_stall_no_mem_rtr", "ls_stall_ldq_full", "ls_stall_storebuf_full",
           "ls_stall_skidbuf_full", "fence_op", "ls_stall_wait_for_data", "ls_op_ret",
           "store_req_wins", "wrbuf_stlh", "wrbuf_stsc", "load_op", "store_op", "vec_load_op", "vec_store_op"],
    # Index 1 is S7 and index 8 is S0 -- the order is reversed in RTL (tt_briscv_pkg.vh:280-293).
    "TTCSR": ["none", "tensix_sstatus7", "tensix_sstatus6", "tensix_sstatus5", "tensix_sstatus4",
              "tensix_sstatus3", "tensix_sstatus2", "tensix_sstatus1", "tensix_sstatus0",
              "tensix_busy", "tensix_queue"],
    "MEM": ["none", "mem_stall_any", "rd_local_reg_ongoing", "rd_mailbox_ongoing", "rd_l1_ongoing",
            "rd_cfg_reg_ongoing", "rd_tensix_reg_ongoing", "rd_tdma_reg_ongoing", "rd_pc_buf_ongoing",
            "local_mem_acc", "tti_push", "tti_push_multi", "l1_rd_req", "l1_rd_locache",
            "icache_acc", "icache_hit"],
}
NAMED_UNIONS = {0x10110111: "stall_union_all"}

# The event that normalises work. tti_push is one Tensix instruction issued by this baby core, so
# "stall cycles per tti_push" is a cost per unit of Tensix work rather than a share of elapsed
# time. That distinction is what made the metric discriminate configs rather than work volume.
WORK_EVENT = "tti_push"


def decode_select(ev: int) -> str:
    if ev == 0:
        return "disabled"
    if ev in NAMED_UNIONS:
        return NAMED_UNIONS[ev]
    parts = []
    for domain, shift in DOMAIN_SHIFTS:
        idx = (ev >> shift) & 0xF
        if idx == 0:
            continue
        cat = EVENTS[domain]
        parts.append(f"{domain.lower()}.{cat[idx] if idx < len(cat) else f'idx{idx}'}")
    if not parts:
        return "disabled"
    return parts[0] if len(parts) == 1 else "union(" + "|".join(parts) + ")"


def keep_steady_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invocation 0, which carries JIT and cache warm-up (runbook 12)."""
    runs = sorted(df.run_id.unique())
    keep = runs[1:] or runs
    return df[df.run_id.isin(keep)]


# ---------------------------------------------------------------------------------------------
def load_tensix(dd: Path, cfg: str, tag: str) -> pd.DataFrame | None:
    """Per-core Tensix metrics for one config, as fractions of that core's elapsed cycles."""
    path = dd / f"{tag}_counters.csv"
    if not path.exists():
        return None
    df = keep_steady_runs(pd.read_csv(path))
    if df.empty:
        return None
    # One value per core per counter, averaged over the kept invocations.
    wide = df.pivot_table(index=["core_x", "core_y"], columns="counter", values="value", aggfunc="mean")
    ref = df.groupby(["core_x", "core_y"]).ref_cnt.mean()
    # A core that never ran a kernel has no elapsed time to normalise by.
    wide = wide[ref > 0]
    ref = ref[ref > 0]

    # Work normaliser. Tensix math instructions issued is proportional to algorithmic work and
    # independent of how long the core took to do it, which is exactly what a cycle-share is not.
    # Every engine utilisation here is a share of the same ref_cnt, so across configs they move in
    # near-perfect lockstep (ratios constant to ~2% on this grid) and cannot discriminate which
    # engine binds. Dividing by work instead of by time breaks that shared denominator.
    work = wide[TENSIX_WORK] if TENSIX_WORK in wide.columns else None
    if work is not None:
        work = work.where(work > 0)

    out = pd.DataFrame(index=wide.index)
    missing = []
    for name, expr in TENSIX_METRICS.items():
        cols = re.findall(r"[A-Z][A-Z0-9_]+", expr)
        if any(c not in wide.columns for c in cols):
            missing.append(name)
            continue
        val = wide.eval(expr)
        out[name] = val / ref                      # share of elapsed cycles (collinear, descriptive)
        if work is not None:
            out[name + "_per_work"] = val / work   # cost per unit of Tensix work (comparable)
    if missing:
        print(f"  note: {cfg} missing Tensix counters for {len(missing)} metrics: {missing[:4]}")

    for name, cols in (("noc_read_bytes_per_cyc", NOC_READ_INTO_L1), ("noc_write_bytes_per_cyc", NOC_WRITE_OUT_OF_L1)):
        if all(c in wide.columns for c in cols):
            out[name] = wide[cols].sum(axis=1) * 16 / ref
            if work is not None:
                out[name.replace("_per_cyc", "") + "_per_work"] = wide[cols].sum(axis=1) * 16 / work
    out["tensix_ref_cycles"] = ref
    if work is not None:
        out["tensix_work"] = work
        # The target variable: inverse throughput. High means the datapath took many cycles to do
        # the same amount of work, i.e. it spent them waiting. This is what the campaign explains.
        out["cycles_per_work"] = ref / work
    out["config"] = cfg
    return out.reset_index()


def load_pmc(dd: Path, cfg: str, tag: str, pass_name: str) -> pd.DataFrame:
    """Per-core, per-RISC event counts for one event pair, long form."""
    path = dd / f"{tag}_riscv_pmc.csv"
    if not path.exists():
        return pd.DataFrame()
    df = keep_steady_runs(pd.read_csv(path))
    rows = []
    for _, r in df.iterrows():
        for ev, value in ((int(r.ev3), int(r.pmc3)), (int(r.ev4), int(r.pmc4))):
            if ev == 0:
                continue
            rows.append({
                "config": cfg, "pass": pass_name, "core_x": int(r.core_x), "core_y": int(r.core_y),
                "risc": r.risc, "event": decode_select(ev), "value": value,
                "mcycle": int(r.mcycle), "minstret": int(r.minstret),
            })
    return pd.DataFrame(rows)


def load_wall(dd: Path, cfg: str, tags: list[str]) -> float:
    """Device wall in cycles: max(KERNEL end) - min(KERNEL start) over all cores and RISCs.

    Taken from a plain capture by preference -- the multipass run replays the workload and its
    wall is not comparable.
    """
    for tag in tags:
        p = dd / f"{tag}_runs.csv"
        if not p.exists():
            continue
        df = keep_steady_runs(pd.read_csv(p))
        if not df.empty:
            return float(df.wall_dev_cycles.mean())
    return float("nan")


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd", default=os.environ.get("DD", "."))
    ap.add_argument("--min-configs", type=int, default=5,
                    help="below this many non-ablation configs a correlation is not worth printing")
    a = ap.parse_args()
    dd = Path(a.dd).resolve()

    # Discover <cfg>__<pass> tags from whatever landed on disk, so a partial campaign still reduces.
    tags = {}
    for p in sorted(dd.glob("*__*_riscv_pmc.csv")):
        stem = p.name[: -len("_riscv_pmc.csv")]
        cfg, _, pass_name = stem.partition("__")
        tags.setdefault(cfg, {})[pass_name] = stem
    if not tags:
        raise SystemExit(f"no <cfg>__<pass>_riscv_pmc.csv captures in {dd}")

    print(f"found {len(tags)} configs, {sum(len(v) for v in tags.values())} captures in {dd}")

    tensix_rows, pmc_rows, walls = [], [], {}
    for cfg, passes in sorted(tags.items()):
        # Exactly one pass per config carries the Tensix counters (the multipass run).
        tx = None
        for pass_name, tag in sorted(passes.items()):
            if tx is None:
                tx = load_tensix(dd, cfg, tag)
            pmc_rows.append(load_pmc(dd, cfg, tag, pass_name))
        if tx is None:
            print(f"  warning: {cfg} has no _counters.csv, Tensix side missing")
        else:
            tensix_rows.append(tx)
        plain = [t for pn, t in sorted(passes.items()) if not (dd / f"{t}_counters.csv").exists()]
        walls[cfg] = load_wall(dd, cfg, plain + list(passes.values()))

    pmc = pd.concat([d for d in pmc_rows if not d.empty], ignore_index=True)
    pmc["frac"] = pmc.value / pmc.mcycle.replace(0, np.nan)
    pmc["ipc"] = pmc.minstret / pmc.mcycle.replace(0, np.nan)

    # Work normaliser: tti_push per core per RISC, from whichever pass measured it. It is armed in
    # more than one pass on purpose, which doubles as a run-to-run stability check.
    work = (pmc[pmc.event.str.endswith("." + WORK_EVENT)]
            .groupby(["config", "core_x", "core_y", "risc"]).value.mean().rename("tti_push"))
    pmc = pmc.merge(work, on=["config", "core_x", "core_y", "risc"], how="left")
    pmc["per_tti"] = pmc.value / pmc.tti_push.replace(0, np.nan)

    # Also normalise by the SAME Tensix work counter that defines cycles_per_work. Only then is
    # the cycle budget an exact decomposition: for one baby core, mcycle/work is the total and
    # each stall/work is a component of it, in identical units. Normalising the RISC-V side by
    # its own tti_push instead leaves an unknown scale factor between the two sides.
    if tensix_rows:
        txw = (pd.concat(tensix_rows, ignore_index=True)[["config", "core_x", "core_y", "tensix_work"]]
               .drop_duplicates(["config", "core_x", "core_y"]))
        pmc = pmc.merge(txw, on=["config", "core_x", "core_y"], how="left")
        pmc["per_work"] = pmc.value / pmc.tensix_work.replace(0, np.nan)
        pmc["mcycle_per_work"] = pmc.mcycle / pmc.tensix_work.replace(0, np.nan)

    pmc.to_csv(dd / "pmc_tensix_long.csv", index=False)

    # -------- collapse to one row per config --------
    # Cross-core mean. The core is not the unit of observation here (see the module docstring);
    # it is averaged over so that each config contributes exactly one point.
    stats = ["frac", "per_tti"] + (["per_work"] if "per_work" in pmc.columns else [])
    pmc_cfg = pmc.pivot_table(index="config", columns=["risc", "event"],
                              values=stats, aggfunc="mean")
    pmc_cfg.columns = [f"{risc}.{ev}.{stat}" for stat, risc, ev in pmc_cfg.columns]
    for col, label in (("ipc", "ipc"), ("mcycle", "mcycle"), ("mcycle_per_work", "mcycle_per_work")):
        if col not in pmc.columns:
            continue
        t = pmc.pivot_table(index="config", columns="risc", values=col, aggfunc="mean")
        t.columns = [f"{c}.{label}" for c in t.columns]
        pmc_cfg = pmc_cfg.join(t)

    tensix = pd.concat(tensix_rows, ignore_index=True) if tensix_rows else pd.DataFrame()
    tx_cfg = tensix.drop(columns=["core_x", "core_y"]).groupby("config").mean(numeric_only=True)

    wide = tx_cfg.join(pmc_cfg, how="outer")
    wide["wall_cycles"] = pd.Series(walls)
    # Config knobs, parsed back out of the tag so the table is self-describing.
    meta = pd.DataFrame(index=wide.index)
    meta["causal"] = [0 if "noncausal" in c else 1 for c in wide.index]
    meta["q_chunk"] = [int(m.group(1)) if (m := re.search(r"q(\d+)k", c)) else np.nan for c in wide.index]
    meta["k_chunk"] = [int(m.group(1)) if (m := re.search(r"k(\d+)$", c)) else np.nan for c in wide.index]
    meta["is_ablation"] = [c.startswith("abl_") for c in wide.index]
    wide = meta.join(wide)
    wide.to_csv(dd / "pmc_tensix_configs.csv")

    # -------- the correlation, across configs, baselines only --------
    base = wide[~wide.is_ablation]
    tx_cols = [c for c in tx_cfg.columns if c in base.columns] + ["wall_cycles"]
    pmc_cols = [c for c in pmc_cfg.columns if c in base.columns]

    rows = []
    if len(base) >= a.min_configs:
        for pc in pmc_cols:
            x = base[pc]
            if x.notna().sum() < a.min_configs or x.std(skipna=True) == 0:
                continue
            risc, _, rest = pc.partition(".")
            event, _, stat = rest.rpartition(".")
            for tc in tx_cols:
                y = base[tc]
                ok = x.notna() & y.notna()
                if ok.sum() < a.min_configs or y[ok].std() == 0 or x[ok].std() == 0:
                    continue
                # Slope of the PMC metric with respect to the Tensix metric. When both are
                # normalised by the same work counter and the Tensix metric is cycles_per_work,
                # this reads directly as "how many of each additional cycle per unit work this
                # stall accounts for" -- a share of the marginal cost. It is the discriminating
                # statistic here, because r is pinned near 1 for every stall by the identity
                # mcycle = stalled + retired.
                slope = float(np.polyfit(y[ok], x[ok], 1)[0])
                rows.append({"risc": risc, "event": event, "stat": stat, "tensix_metric": tc,
                             "r": float(np.corrcoef(x[ok], y[ok])[0, 1]), "slope": slope,
                             "n": int(ok.sum())})
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["abs_r"] = corr.r.abs()
        corr = corr.sort_values("abs_r", ascending=False)
        corr.to_csv(dd / "pmc_tensix_corr.csv", index=False)

    # How much of one measurement the engine cycle-shares really are. If the first principal
    # component explains almost everything, then "which engine is the bottleneck" has no answer
    # from these columns and the report must say so rather than rank near-identical correlations.
    collin = collinearity(base, [c for c in HEADLINE_TENSIX if c in base.columns])

    # -------- ablations as deltas against their baseline --------
    abl_rows = []
    for cfg in wide.index[wide.is_ablation]:
        m = re.match(r"abl_(\w+?)_((?:non)?causal_q\d+k\d+)$", cfg)
        if not m:
            continue
        baseline = f"t21_{m.group(2)}"
        if baseline not in wide.index:
            continue
        for col in tx_cols + pmc_cols:
            b, v = wide.at[baseline, col], wide.at[cfg, col]
            if pd.notna(b) and pd.notna(v) and b != 0:
                abl_rows.append({"ablation": m.group(1), "baseline": baseline, "metric": col,
                                 "base": b, "ablated": v, "delta_pct": 100 * (v / b - 1)})
    abl = pd.DataFrame(abl_rows)
    if not abl.empty:
        abl.to_csv(dd / "pmc_tensix_ablation.csv", index=False)

    write_report(dd, wide, corr, abl, base, collin)
    print(f"\nwrote pmc_tensix_{{configs,long,corr,ablation}}.csv and pmc_tensix_report.md in {dd}")


def collinearity(base: pd.DataFrame, cols: list[str]) -> dict:
    """Share of variance in the engine cycle-shares explained by their first principal component."""
    sub = base[cols].dropna(axis=1, how="all").dropna()
    if len(sub) < 3 or sub.shape[1] < 2:
        return {}
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1)
    if z.shape[1] < 2:
        return {}
    ev = np.linalg.svd(z.values, compute_uv=False) ** 2
    pairs = sub.corr()
    off = pairs.where(~np.eye(len(pairs), dtype=bool))
    # Ratios between engine utilisations: if these are constant across configs, the engines are
    # not independently observable and no single one can be named as the binding lane.
    ratios = {}
    if "fpu_util" in sub.columns:
        for c in sub.columns:
            if c != "fpu_util" and (sub["fpu_util"] > 0).all():
                r = sub[c] / sub["fpu_util"]
                ratios[c] = (float(r.mean()), float(100 * r.std() / r.mean()) if r.mean() else np.nan)
    return {"pc1_frac": float(ev[0] / ev.sum()), "min_abs_r": float(off.abs().min().min()),
            "cols": list(sub.columns), "n": len(sub), "ratios_vs_fpu": ratios}


def write_report(dd, wide, corr, abl, base, collin):
    L = []
    L.append("# RISC-V PMC against Tensix counters\n")
    L.append(f"Configs: {len(wide)} total, {len(base)} baseline, {int(wide.is_ablation.sum())} ablation. "
             f"Correlations are across the {len(base)} baseline configs; each config contributes one "
             "point, being the cross-core mean. Correlating across cores at one config is an "
             "accounting identity and is deliberately not done here.\n")

    show = [c for c in ["causal", "q_chunk", "k_chunk", "wall_cycles"] + HEADLINE_TENSIX if c in wide.columns]
    L.append("\n## Tensix engine utilisation and stalls, by config\n")
    L.append("Fractions of elapsed core cycles (`ref_cnt`), averaged over active cores.\n")
    L.append(wide.loc[~wide.is_ablation, show].round(4).to_markdown())

    # ---- the caveat that governs how everything below may be read ----
    if collin:
        L.append("\n\n## These engine utilisations are one measurement, not six\n")
        pc1 = 100 * collin["pc1_frac"]
        L.append(f"Across the {collin['n']} baseline configs the first principal component of "
                 f"{HEADLINE_TENSIX} explains **{pc1:.2f}%** of their variance, and the weakest "
                 f"pairwise correlation among them is {collin['min_abs_r']:.4f}. Their ratios to "
                 "`fpu_util` are near-constant:\n")
        rt = pd.DataFrame([{"metric": k, "mean ratio to fpu_util": round(v[0], 4), "cv %": round(v[1], 2)}
                           for k, v in collin["ratios_vs_fpu"].items()])
        L.append(rt.to_markdown(index=False))
        L.append("\n\nThis is the same accounting identity that defeats the cross-core analysis, "
                 "reappearing across configs: every one of these is a share of the same `ref_cnt`, "
                 "and the kernel does a fixed amount of engine work per unit of algorithmic work, "
                 "so what varies between configs is mostly how long the core took. The coefficients "
                 "of variation above show how far that holds: whatever does not load on the first "
                 "component is a genuine change in the work mix (read it from the `*_per_work` "
                 "table), not a change in which engine binds. Ranking the cycle-share "
                 "correlations against anything mostly ranks the same number several times.\n")
        L.append("\nThe consequence is that the interesting question is not *which engine is "
                 "saturated* but *why the whole datapath is idle*, and the quantity to explain is "
                 "`cycles_per_work` = elapsed cycles per Tensix math instruction issued. That is "
                 "the target variable used below. Engine counters divided by the same work "
                 "normaliser (`*_per_work`) no longer share a denominator with it.\n")

    wcols = [c for c in HEADLINE_WORK if c in wide.columns]
    if wcols:
        L.append("\n## Work-normalised view\n")
        L.append("Cycles and engine-busy cycles per Tensix math instruction issued. "
                 "`cycles_per_work` is inverse throughput: higher means the datapath spent more "
                 "time to do the same work.\n")
        L.append(wide.loc[~wide.is_ablation, ["causal", "q_chunk", "k_chunk"] + wcols]
                 .round(4).to_markdown())

    if not corr.empty:
        L.append("\n\n## Where the marginal cycle goes\n")
        L.append("`mcycle = stalled + retired` holds to about 1% on this single-issue pipeline, "
                 "so **every** stall correlates with `cycles_per_work` at r close to 1 by "
                 "construction. Correlation therefore cannot rank them. The discriminating "
                 "statistic is the **slope**: with both sides divided by the same work counter, "
                 "`slope` is how many of each additional cycle-per-unit-work that stall accounts "
                 "for. A slope near 1 means that stall absorbs essentially the whole marginal "
                 "cost; a slope near 0 means the stall is fixed overhead that does not grow as "
                 "the config gets worse.\n")
        sub = corr[(corr.tensix_metric == "cycles_per_work") & (corr.stat == "per_work")]
        if not sub.empty:
            top = (sub[sub.event == "stall_union_all"].set_index("risc").slope
                   .reindex(["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]).dropna())
            if not top.empty:
                L.append("\n### Which baby core absorbs the marginal cycle\n")
                L.append("Slope of that core's total stall against `cycles_per_work`. The three "
                         "TRISCs are, in order, the unpack, math and pack threads.\n")
                lbl = {"TRISC_0": "TRISC_0 (unpack)", "TRISC_1": "TRISC_1 (math)",
                       "TRISC_2": "TRISC_2 (pack)", "NCRISC": "NCRISC (reader)", "BRISC": "BRISC (writer)"}
                L.append(pd.DataFrame({"thread": [lbl.get(i, i) for i in top.index],
                                       "slope of stall_union vs cycles_per_work": top.round(4).values}
                                      ).to_markdown(index=False))
            L.append("\n\n### Slope against `cycles_per_work`, per-work normalised\n")
            L.append(sub.sort_values("slope", ascending=False)
                     .head(20)[["risc", "event", "slope", "r", "n"]].round(4).to_markdown(index=False))
        L.append("\n\n### Ranked by correlation (shown to make the identity visible, not to rank)\n")
        for tm in ["cycles_per_work", "wall_cycles"]:
            s2 = corr[corr.tensix_metric == tm].head(10)
            if s2.empty:
                continue
            L.append(f"\n#### vs `{tm}`\n")
            L.append(s2[["risc", "event", "stat", "r", "slope", "n"]].round(4).to_markdown(index=False))

        L.append("\n\n## Per engine metric (collinear -- read the caveat above)\n")
        for tm in HEADLINE_TENSIX:
            sub = corr[corr.tensix_metric == tm].head(5)
            if sub.empty:
                continue
            L.append(f"\n### vs `{tm}`\n")
            L.append(sub[["risc", "event", "stat", "r", "n"]].round(4).to_markdown(index=False))

    if not abl.empty:
        L.append("\n\n## Ablations\n")
        L.append("Percent change against the same shape with the lane intact. A metric that "
                 "collapses when a lane is removed is caused by that lane; one that does not move "
                 "is merely correlated with it.\n")
        for name, grp in abl.groupby("ablation"):
            key = grp[grp.metric.isin(HEADLINE_TENSIX + HEADLINE_WORK + ["wall_cycles"])]
            if key.empty:
                continue
            L.append(f"\n### `{name}`\n")
            L.append(key.pivot_table(index="metric", columns="baseline", values="delta_pct")
                     .round(2).to_markdown())

    (dd / "pmc_tensix_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
