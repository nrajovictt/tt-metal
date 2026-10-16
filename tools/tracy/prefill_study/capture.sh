#!/bin/bash
# Profile one prefill length of Llama 3.1 8B (1 layer) on one P150.
#
# usage (from the repo root):
#   tools/tracy/prefill_study/capture.sh counters 4k    # perf counters (FPU/SFPU/MATH)  -> generated/profiler/prefill_4k_1dev
#   tools/tracy/prefill_study/capture.sh zones 4k       # SDPA stage zones (apply sdpa_stage_zones.patch first!)
#                                                       #   -> generated/profiler/sdpa_zones/prefill_4k
# LENGTH is a test id: 128 256 512 1k 2k 3k 4k 8k 16k 32k
#
# Runs one at a time (single device). 32k takes ~7 min, mostly the CPU reference model.
set -u
MODE=$1
LEN=$2
HERE=$(dirname "$0")
source "$HERE/env.sh"

case $MODE in
  counters) D=generated/profiler/prefill_${LEN}_1dev; EXTRA="--profiler-capture-perf-counters=fpu" ;;
  zones)
    if ! grep -q 'DeviceZoneScopedN("SDPA-QK")' ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp; then
      echo "SDPA zones not applied: git apply $HERE/sdpa_stage_zones.patch" >&2; exit 1
    fi
    D=generated/profiler/sdpa_zones/prefill_${LEN}; EXTRA="" ;;
  *) echo "mode must be counters or zones" >&2; exit 1 ;;
esac

rm -rf "$D"; mkdir -p "$D"
# 'not accuracy' matters: the word 'performance' also matches the test's models_performance_bare_metal marker,
# so '-k performance' alone selects the accuracy variant too.
python -m tracy -r -p -v $EXTRA --op-support-count=100000 -o "$D" \
  -m "pytest models/tt_transformers/tests/test_model_prefill.py -k '1layer and not accuracy and max128k-${LEN}-'" > "$D/run.log" 2>&1
RC=$?
echo "$LEN exit=$RC $(grep -o 'Prefill seq len: [0-9]*' "$D/run.log" | tail -1)" \
     "$(grep -o 'Output PCC: [0-9.]*' "$D/run.log" | tail -1)" \
     "$(grep -oE '[0-9]+ (passed|failed)' "$D/run.log" | tail -1)" \
     "dropped-marker warnings=$(grep -c 'markers were dropped' "$D/run.log")"
