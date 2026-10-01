#!/bin/bash
# RISC-V baby-core HPM stall sweep. Every pass instruments all five baby cores at once
# (BRISC, NCRISC, TRISC0/1/2) because each core programs and reads its own CSRs.
#
# Why a sweep rather than one run: the hardware has exactly two programmable counters per core, and
# the selected events are OR-ed rather than summed, so one run yields two event streams per core.
# mcycle and minstret ride along free in every pass, which is what makes the passes comparable --
# every count is normalised by the mcycle of its own run. See BRISC_PERF_COUNTERS.md sections 4-7.
#
# Each pass changes the compile-time event selects, which changes the JIT cache key and forces a
# firmware rebuild. That is expected and is why a pass costs a full run.
#
# Usage: ./campaign_pmc.sh [tag_prefix]
#   Config comes from the same SDPA_* env vars run_zone.sh takes.
#   PASSES=... restricts to a subset, e.g. PASSES="stall_union tensix" ./campaign_pmc.sh
set -e
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
PREFIX=${1:-pmc}

# name | mhpmevent3 | mhpmevent4.  Nibble order low-to-high is BP, IF, DE, DE1, EX, LS, TTCSR, MEM;
# index 0 in a domain is the tie-off, so a zero nibble disables that domain.
PASS_LIST=(
  # Top-line: union of the six per-stage StallAny events, i.e. cycles where at least one stage
  # could not advance. Not obtainable by summing the per-stage counts. Paired with the BP stall.
  "stall_union|0x10110111|0x00000001"
  # Per-stage stall decomposition.
  "if_de|0x00000010|0x00000100"
  "ex_ls|0x00010000|0x00100000"
  # MEM stall plus the Tensix instruction-buffer write rate. On a TRISC these two together are the
  # TTI backpressure signature: a blocked TTI push is a store that was not granted.
  "mem_tti|0x10000000|0xA0000000"
  # Blocked on Tensix. These mirror the live Tensix status wires into the core's CSR file, so they
  # are the only direct view of Tensix state from the RISC-V side: busy = some unit working,
  # queue = instruction queue non-empty. A core with high tensix_busy and high mem_stall is blocked
  # by Tensix; high mem_stall with low tensix_busy is L1 or NOC contention instead.
  "tensix|0x09000000|0x0A000000"
  # Store-buffer-full stall (the TTI backpressure mechanism) against the serialization stall, which
  # is what our own four CSR reads per launch register as and therefore needs subtracting.
  "backpress|0x00400000|0x00000400"
  # Instruction supply.
  "icache|0xE0000000|0xF0000000"
  # L1 read traffic and how much of it the low cache absorbed.
  "l1|0xC0000000|0xD0000000"
  # Scalar load and store rates, for an independent check on the LS stall fractions.
  "ldst|0x00C00000|0x00D00000"
)

WANT=${PASSES:-}
TAGS=()
for entry in "${PASS_LIST[@]}"; do
  NAME=${entry%%|*}; REST=${entry#*|}; EV3=${REST%%|*}; EV4=${REST#*|}
  if [ -n "$WANT" ] && ! echo " $WANT " | grep -q " $NAME "; then continue; fi
  TAG=${PREFIX}_${NAME}
  echo "=== pass $NAME  mhpmevent3=$EV3 mhpmevent4=$EV4 -> $TAG"
  TT_METAL_PROFILE_RISCV_PMC="$EV3,$EV4" "$SD/run_zone.sh" "$TAG" plain
  TAGS+=("$TAG")
done

if [ ${#TAGS[@]} -eq 0 ]; then echo "no passes selected (PASSES=\"$WANT\")"; exit 1; fi

echo "=== reducing ${#TAGS[@]} passes"
cd $TTM; source $PYENV
python "$SD/riscv_pmc_table.py" "${TAGS[@]}" \
  --config "${PREFIX} S=${SDPA_SEQ:-4096} q=${SDPA_QCHUNK:-128} k=${SDPA_KCHUNK:-${SDPA_QCHUNK:-128}} causal=${SDPA_CAUSAL:-1}" \
  --out-prefix "${PREFIX}"
echo "=== $PREFIX done $(date -u +%H:%M:%SZ)"
