# Source from the repo root before any device run:  source tools/tracy/prefill_study/env.sh
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export MESH_DEVICE=P150
# 32 -> max prefill chunk 32768 tokens; without it the chunk size defaults to 4096
export MAX_PREFILL_CHUNK_SIZE=32
# Test-only switch in test_model_prefill.py: prefill at the prompt length itself instead of the next
# power-of-2 bucket (512 stays 512, 3072 stays 3072). Power-of-2 lengths are unaffected.
export TT_PREFILL_MIN_PADDING=1
source python_env/bin/activate
