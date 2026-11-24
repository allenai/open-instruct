#!/bin/bash
# Run inside the committed MILES image on two B300s. The checkpoint must fit
# TP1 for the sequential TP1/TP2 greedy comparison (Small does at short context).
set -euo pipefail
: "${1:?Usage: bash scripts/miles/tp2_serving_smoke.sh HF_CHECKPOINT}"
cd /opt/core-rl/sources/olmo-sglang
python -m pytest tests/test_rounding.py tests/test_core_compat.py -q
export OLMO_SGLANG_CORE_COMPAT=rounding
python -m olmo_sglang.validation.tensor_parallel \
    --model "$1" --tp-size 2 --prompt-length 32 --max-new-tokens 16 \
    --context-length 128 --mem-fraction-static 0.8
