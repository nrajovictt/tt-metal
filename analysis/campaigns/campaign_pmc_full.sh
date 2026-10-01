#!/bin/bash
# Full counter campaign: all 155 Blackhole Tensix perf counters plus the complete RISC-V baby-core
# HPM event sweep, over a config grid, so that the two can be correlated ACROSS CONFIGS.
#
# Why it is shaped this way
# -------------------------
# The RISC-V side has exactly two programmable counters per baby core, so covering the event
# catalog costs one run per event pair (PASS_LIST below). The Tensix side needs --perf-counter-
# multipass, which replays the workload five times and costs ~5.5 min against ~1 min for a plain
# run. Capturing Tensix once per event pair would therefore cost 5.5 min x passes x configs.
#
# It is not necessary. The Tensix counters are a property of the config, not of the RISC-V event
# selects: across three same-config mp captures with different mhpmevent pairs, all 155 counters
# agree to a median of 0.004% and a worst case of 0.36% (verified 2026-09-22 on the anchor, from
# the pmc_combo / pmc_mp_backpress / pmc_mp_exls captures). So each config gets exactly one mp run,
# which carries the Tensix counters and the headline PMC pair together, and the remaining pairs
# ride cheap plain runs. Per config that is ~5.5 + (N-1) min instead of ~5.5N.
#
# Everything runs ZONES OFF. Zone markers perturb the very stalls the PMC is measuring, and the
# zone tax is itself config-dependent (CONTEXT.md section 6), which would contaminate a
# cross-config correlation. The ablation configs set their toggle with zones still off.
#
# Correlate across configs, never across cores at one config: two per-cycle fractions sharing a
# denominator correlate at |r| ~ 1 by accounting identity, not by mechanism
# (BRISC_PERF_COUNTERS.md, "Do not correlate per-cycle fractions across cores").
#
# Usage: ./campaign_pmc_full.sh
#   DD=<dir>          where captures land (must be outside the checkout)
#   PASSES="a b"      restrict the event sweep to named passes
#   CONFIGS_ONLY="x"  restrict to named configs
#   SKIP_ABL=1        drop the ablation block
#   FORCE=1           re-run tags that already have output (default is resume)
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
cd $DD

# ---------------------------------------------------------------------------------------------
# RISC-V HPM event pairs. name | mhpmevent3 | mhpmevent4
# Nibble order low-to-high is BP, IF, DE, DE1, EX, LS, TTCSR, MEM; index 0 in a domain is the
# hardwired tie-off, so a zero nibble disables that domain. Catalog: BRISC_PERF_COUNTERS.md S5.
# The increment is an OR across domains, never a sum, so a multi-domain select measures the union.
PASS_LIST=(
  # Headline. Union of the six per-stage StallAny events against the Tensix instruction push rate,
  # so that "stall cycles per unit of Tensix work" is formed within a single pass. Same pair as the
  # 2026-09-18 T2.1 capture, which keeps this campaign directly comparable to it.
  "stall_union|0x10110111|0xA0000000"
  # Per-stage stall decomposition.
  "if_de|0x00000010|0x00000100"
  # ex_stall_no_instr, NOT ex_stall_any: index 1 reads identically zero on this pipeline because
  # hazards and unit-readiness resolve at decode (BRISC_PERF_COUNTERS.md S5, EX). Index 2 is EX
  # starved of work, which is the informative one. Paired with the LS aggregate stall.
  "ex_ls|0x00020000|0x00100000"
  # MEM stall plus TTI push rate: on a TRISC this pair is the Tensix-backpressure signature, since
  # a blocked Tensix instruction issue is a store that was not granted (S7.1).
  "mem_tti|0x10000000|0xA0000000"
  # Store-buffer-full stall (the mechanism behind the above) against the serialization stall, which
  # our own four CSR reads per launch register as and which therefore needs subtracting.
  "backpress|0x00400000|0x00000400"
  # NEW. Load waiting on returning data, against cycles with an L1 read outstanding. This is the
  # L1/NoC latency signature, distinct from "could not issue" back-pressure above.
  "waitdata|0x00700000|0x40000000"
  # NEW. Splits decode stalls into a real RAW hazard and simple instruction starvation.
  "hazard|0x00000500|0x00000300"
  # NEW. Tensix instruction issue seen at decode, cross-checking the MEM-side tti_push, against
  # branch mispredicts as a control-flow cost.
  "issue|0x0000D000|0x00030000"
  # Instruction supply.
  "icache|0xE0000000|0xF0000000"
  # Scalar load and store rates, an independent check on the LS stall fractions.
  "ldst|0x00C00000|0x00D00000"
  # The two Tensix-facing read paths: the PC buffer (how a baby core waits on Tensix sync) and
  # Tensix register reads. With the whole TTCSR domain measured dead (below), these are the only
  # remaining direct view of a core blocking on Tensix state.
  "tensix_rd|0x80000000|0x60000000"
  # Decomposes the part of ls_stall_any that is not store-buffer-full. On TRISC_1 at the anchor
  # ls_stall_any is 0.690 while ls_stall_storebuf_full is 0.326, so a third of the largest stall
  # on the math thread is otherwise unattributed. Load-queue full and skid-buffer full are the
  # two remaining mechanisms.
  "lsq|0x00300000|0x00500000"
)
# Measured dead on this silicon at the anchor, 2026-09-22, and therefore not swept:
#   tensix_sem|0x08000000|0x07000000  ttcsr.tensix_sstatus0 / sstatus1 -- both exactly 0 on all
#     five cores. The narrower TTCSR bits do not escape the problem that makes tensix_busy and
#     tensix_queue useless (BRISC_PERF_COUNTERS.md S7.3 left this open); the domain is unusable.
#   l1|0xC0000000|0xD0000000  mem.l1_rd_req peaks at 1e-4 and mem.l1_rd_locache is flat zero,
#     while mem.rd_l1_ongoing in the waitdata pass clearly shows L1 reads happening. MEM indices
#     12/13 do not observe this core's L1 traffic.
# Single-config captures of both are kept in $DD as the evidence.
# Which pass rides the expensive multipass run that also yields the Tensix counters.
MP_PASS=${MP_PASS:-stall_union}

# ---------------------------------------------------------------------------------------------
# Configs. name | causal | q_chunk | k_chunk | reader_stub | exp_stub
# The T2.1 grid is the canonical seven (q,k) points crossed with causality.
CONFIG_LIST=()
POINTS=${POINTS:-"128:128 64:128 256:128 128:256 128:512 512:128 512:512"}
MODES=${MODES:-"causal noncausal"}
for QK in $POINTS; do
  QC=${QK%%:*}; KC=${QK##*:}
  for MODE in $MODES; do
    C=1; [ $MODE = noncausal ] && C=0
    CONFIG_LIST+=("t21_${MODE}_q${QC}k${KC}|$C|$QC|$KC|0|0")
  done
done
# Ablations, at the two shapes that bind on different lanes: causal q128 k128 is DRAM-bound and
# non-causal q512 k512 is PACK-bound (CONTEXT.md, 2026-09-18). Removing a lane turns a correlation
# into an attribution test -- if NoC reads gate the wall, READER_STUB should collapse the stalls
# that correlate with it.
if [ "${SKIP_ABL:-0}" != 1 ]; then
  CONFIG_LIST+=("abl_readerstub_causal_q128k128|1|128|128|1|0")
  CONFIG_LIST+=("abl_expstub_causal_q128k128|1|128|128|0|1")
  CONFIG_LIST+=("abl_readerstub_noncausal_q512k512|0|512|512|1|0")
  CONFIG_LIST+=("abl_expstub_noncausal_q512k512|0|512|512|0|1")
fi

# ---------------------------------------------------------------------------------------------
WANT_PASS=${PASSES:-}
WANT_CFG=${CONFIGS_ONLY:-}
selected_cfgs=(); for e in "${CONFIG_LIST[@]}"; do
  n=${e%%|*}; [ -n "$WANT_CFG" ] && ! echo " $WANT_CFG " | grep -q " $n " && continue
  selected_cfgs+=("$e")
done
selected_passes=(); for e in "${PASS_LIST[@]}"; do
  n=${e%%|*}; [ -n "$WANT_PASS" ] && ! echo " $WANT_PASS " | grep -q " $n " && continue
  selected_passes+=("$e")
done
NCFG=${#selected_cfgs[@]}; NPASS=${#selected_passes[@]}
TOTAL=$((NCFG * NPASS))
[ $TOTAL -eq 0 ] && { echo "nothing selected"; exit 1; }

# Per-config cost: one mp run plus the rest plain. Measured on the p100a, 2026-09-18.
EST=$(( NCFG * 330 + (TOTAL - NCFG) * 65 ))
echo "=== campaign_pmc_full: $NCFG configs x $NPASS passes = $TOTAL runs"
echo "=== DD=$DD  card=$CARD  est $((EST/3600))h$(( (EST%3600)/60 ))m"
echo "=== mp pass (Tensix counters ride along): $MP_PASS"
START=$(date +%s); DONE=0; FAILED=()

for centry in "${selected_cfgs[@]}"; do
  IFS='|' read -r CNAME CAUSAL QC KC RS ES <<< "$centry"
  # Zones off always; the ablation toggle is the only thing that varies.
  $SD/set_zone_config.sh 0 "$RS" 0 "$ES" 0 >/dev/null
  for pentry in "${selected_passes[@]}"; do
    IFS='|' read -r PNAME EV3 EV4 <<< "$pentry"
    MODE=plain; [ "$PNAME" = "$MP_PASS" ] && MODE=mp
    TAG="${CNAME}__${PNAME}"
    DONE=$((DONE + 1))
    if [ "${FORCE:-0}" != 1 ] && [ -s "$DD/${TAG}_riscv_pmc.csv" ] \
       && { [ "$MODE" != mp ] || [ -s "$DD/${TAG}_counters.csv" ]; }; then
      echo "--- [$DONE/$TOTAL] $TAG already captured, skipping"
      continue
    fi
    ELAPSED=$(( $(date +%s) - START ))
    echo "--- [$DONE/$TOTAL] $TAG mode=$MODE ev3=$EV3 ev4=$EV4 elapsed=$((ELAPSED/60))m"
    if ! SDPA_CAUSAL=$CAUSAL SDPA_QCHUNK=$QC SDPA_KCHUNK=$KC \
         TT_METAL_PROFILE_RISCV_PMC="$EV3,$EV4" $SD/run_zone.sh "$TAG" "$MODE" \
         < /dev/null; then
      echo "!!! FAILED $TAG"; FAILED+=("$TAG")
    fi
  done
done

$SD/set_zone_config.sh 0 0 0 0 0 >/dev/null
echo "=== capture done in $(( ($(date +%s) - START) / 60 ))m, ${#FAILED[@]} failures"
[ ${#FAILED[@]} -gt 0 ] && printf '!!! %s\n' "${FAILED[@]}"

cd $TTM; source $PYENV
python "$SD/pmc_tensix_corr.py" || echo "note: reduction failed, captures are intact in $DD"
echo "=== campaign_pmc_full done $(date -u +%H:%M:%SZ)"
