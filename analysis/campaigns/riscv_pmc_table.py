# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduce RISC-V baby-core HPM captures into per-core stall tables for the model.

Inputs: one or more <tag>_riscv_pmc.csv produced by analysis/zone_reduce.py, i.e. one per pass of
the event sweep (see campaign_pmc.sh). Each row carries the two counter deltas for one kernel
launch on one baby core, plus mcycle, minstret, and the mhpmevent selects that produced them.

Event names are decoded from the selects that are in the capture itself, not from the tag, so a
mislabelled or reordered pass cannot produce a mislabelled column. See BRISC_PERF_COUNTERS.md
sections 4 and 5 for the encoding and the full catalog.

Outputs (in $DD):
  riscv_pmc_long.csv     one row per tag/run/core/risc/counter: raw value, mcycle, fraction
  riscv_pmc_by_core.csv  one row per core/risc, one column per event fraction, merged over passes
  riscv_pmc_by_risc.csv  the same collapsed to one row per risc (mean over active cores)

Usage: python riscv_pmc_table.py <tag> [<tag> ...] [--config LABEL] [--out-prefix riscv_pmc]
"""

import argparse
import os
from pathlib import Path

import pandas as pd

DD = Path(os.environ.get("DD", ".")).resolve()

# Nibble order within mhpmevent, low to high. Index 0 of every domain is the hardwired *PmcNone
# tie-off, so a zero nibble means the domain contributes nothing.
DOMAIN_SHIFTS = [
    ("BP", 0),
    ("IF", 4),
    ("DE", 8),
    ("DE1", 12),
    ("EX", 16),
    ("LS", 20),
    ("TTCSR", 24),
    ("MEM", 28),
]

# Event catalog, transcribed from ws-tensix src/hardware/briscv/rtl/tt_briscv_pkg.vh lines 185-312.
# Index position in each list is the nibble value. None marks a bit tied to zero in RTL.
EVENTS = {
    "BP": [
        "none", "pc_stall_any", "bp_stall_no_id_rtr", "bp_stall_no_mem_rtr", "bp_stall_pc_fifo_full",
        "bp_stall_bp_fifo_full", "btb_hit", "pred_jalr", "pred_jal", "pred_br",
    ],
    "IF": [
        "none", "if_stall_any", "if_stall_no_id_rtr", "if_stall_lq_full", "if_stall_lq_sb_full",
        "req_stall_any", "req_stall_no_mem_rtr",
    ],
    "DE": [
        "none", "de_stall_any", "de_stall_no_unit_rtr", "de_stall_if_empty", "de_stall_sync",
        "de_stall_hazard", "hazard_int", "hazard_fp", "hazard_vec", "reg_data_forward",
        "mem_disp_ld", "mem_disp_st", "mem_disp_vec_ld", "mem_disp_vec_st", "ex_disp_mul",
        "ex_disp_div",
    ],
    "DE1": [
        "none", "ex_disp_csr", "ex_disp_i", "ex_disp_m", "ex_disp_a", "ex_disp_zb", "fp_disp_s",
        "fp_disp_zfh", "vec_disp_vset", "vec_disp_int", "vec_disp_fixed", "vec_disp_float",
        "vec_disp_ldst", "tti_issue", "vec_replay", "illegal_op",
    ],
    "EX": [
        "none", "ex_stall_any", "ex_stall_no_instr", "mispred_any", "mispred_new_br", "mispred_dir",
        "mispred_tgt_cond", "mispred_tgt_uncond", "inst_ret",
    ],
    "LS": [
        "none", "ls_stall_any", "ls_stall_no_mem_rtr", "ls_stall_ldq_full", "ls_stall_storebuf_full",
        "ls_stall_skidbuf_full", "fence_op", "ls_stall_wait_for_data", "ls_op_ret",
        "store_req_wins", "wrbuf_stlh", "wrbuf_stsc", "load_op", "store_op", "vec_load_op",
        "vec_store_op",
    ],
    # Note the reversal: index 1 is S7 and index 8 is S0 (tt_briscv_pkg.vh:280-293).
    "TTCSR": [
        "none", "tensix_sstatus7", "tensix_sstatus6", "tensix_sstatus5", "tensix_sstatus4",
        "tensix_sstatus3", "tensix_sstatus2", "tensix_sstatus1", "tensix_sstatus0", "tensix_busy",
        "tensix_queue",
    ],
    "MEM": [
        "none", "mem_stall_any", "rd_local_reg_ongoing", "rd_mailbox_ongoing", "rd_l1_ongoing",
        "rd_cfg_reg_ongoing", "rd_tensix_reg_ongoing", "rd_tdma_reg_ongoing", "rd_pc_buf_ongoing",
        "local_mem_acc", "tti_push", "tti_push_multi", "l1_rd_req", "l1_rd_locache", "icache_acc",
        "icache_hit",
    ],
}


# Composite selects that campaign_pmc.sh arms deliberately. The generic union name is accurate but
# unusable as a column header, so the ones we actually sweep get a short alias.
NAMED_UNIONS = {
    0x10110111: "stall_union_all",  # pc/if/de/ex/ls/mem StallAny OR-ed together
}


def decode_select(ev: int) -> str:
    """Name the event(s) a single mhpmevent select arms.

    A select can arm up to eight events at once, one per domain. The RTL ORs them before the
    increment, so the counter measures the union: cycles in which at least one fired. That is a
    meaningful quantity (total stall cycles with no double counting) but it is not a sum, so the
    name has to say so.
    """
    if ev == 0:
        return "disabled"
    if ev in NAMED_UNIONS:
        return NAMED_UNIONS[ev]
    parts = []
    for domain, shift in DOMAIN_SHIFTS:
        idx = (ev >> shift) & 0xF
        if idx == 0:
            continue
        catalog = EVENTS[domain]
        name = catalog[idx] if idx < len(catalog) else f"idx{idx}"
        parts.append(f"{domain.lower()}.{name}")
    if not parts:
        return "disabled"
    if len(parts) == 1:
        return parts[0]
    return "union(" + "|".join(parts) + ")"


def load_tag(tag: str) -> pd.DataFrame:
    """Long rows for one sweep pass, iteration 0 discarded per the campaign convention."""
    path = DD / f"{tag}_riscv_pmc.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    # Campaign convention (runbook 12): 3 invocations per capture, the first is discarded because
    # it carries JIT and cache warm-up.
    keep = sorted(df.run_id.unique())[1:]
    if not keep:
        keep = sorted(df.run_id.unique())
    df = df[df.run_id.isin(keep)]

    rows = []
    for _, r in df.iterrows():
        for counter, ev, value in (("pmc3", int(r.ev3), int(r.pmc3)), ("pmc4", int(r.ev4), int(r.pmc4))):
            if ev == 0:
                continue
            rows.append(
                {
                    "tag": tag,
                    "run_id": int(r.run_id),
                    "launch": int(r.launch),
                    "core_x": int(r.core_x),
                    "core_y": int(r.core_y),
                    "risc": r.risc,
                    "counter": counter,
                    "ev": f"0x{ev:08X}",
                    "event": decode_select(ev),
                    "value": value,
                    "mcycle": int(r.mcycle),
                    "minstret": int(r.minstret),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        # Every count is only meaningful against the cycles of its own run: run-to-run cycle counts
        # differ, so fractions are the only thing comparable across passes.
        out["frac"] = out.value / out.mcycle.replace(0, pd.NA)
        out["ipc"] = out.minstret / out.mcycle.replace(0, pd.NA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="capture tags, one per sweep pass")
    ap.add_argument("--config", default="", help="config label carried into the output")
    ap.add_argument("--out-prefix", default="riscv_pmc")
    a = ap.parse_args()

    long = pd.concat([load_tag(t) for t in a.tags], ignore_index=True)
    if long.empty:
        raise SystemExit(
            f"no riscv_pmc rows found in {DD} for tags {a.tags}. "
            "Was TT_METAL_PROFILE_RISCV_PMC set for the capture?"
        )
    if a.config:
        long["config"] = a.config

    p_long = DD / f"{a.out_prefix}_long.csv"
    long.to_csv(p_long, index=False)

    # Per core/risc, one column per event. Mean over launches and passes; an event measured in more
    # than one pass is averaged, which is also a cheap consistency check on run-to-run stability.
    by_core = long.pivot_table(
        index=["core_x", "core_y", "risc"], columns="event", values="frac", aggfunc="mean"
    ).reset_index()
    denom = long.groupby(["core_x", "core_y", "risc"]).agg(
        mcycle_mean=("mcycle", "mean"), ipc_mean=("ipc", "mean"), n_samples=("value", "size")
    ).reset_index()
    by_core = by_core.merge(denom, on=["core_x", "core_y", "risc"])
    if a.config:
        by_core["config"] = a.config
    p_core = DD / f"{a.out_prefix}_by_core.csv"
    by_core.to_csv(p_core, index=False)

    # Collapsed to one row per risc. Cores with mcycle == 0 never ran a kernel and would drag the
    # means toward zero, so they are dropped rather than averaged in.
    active = by_core[by_core.mcycle_mean > 0]
    by_risc = active.drop(columns=["core_x", "core_y"]).groupby("risc").mean(numeric_only=True).reset_index()
    by_risc["n_cores"] = active.groupby("risc").size().values
    if a.config:
        by_risc["config"] = a.config
    p_risc = DD / f"{a.out_prefix}_by_risc.csv"
    by_risc.to_csv(p_risc, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(by_risc.to_string(index=False))
    print()
    print("wrote", p_long.name, p_core.name, p_risc.name, f"in {DD}")


if __name__ == "__main__":
    main()
