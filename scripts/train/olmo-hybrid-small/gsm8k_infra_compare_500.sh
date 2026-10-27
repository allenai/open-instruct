#!/bin/bash
set -euo pipefail

# Launch once per MODEL_SIZE through scripts/train/build_image_and_launch.sh.
BEAKER_IMAGE="${1:?Expected committed Beaker image}"
shift
MODEL_SIZE="${MODEL_SIZE:?Set MODEL_SIZE to 275m or 450m}"
case "$MODEL_SIZE" in
    275m|450m) ;;
    *) echo "Unsupported comparison size: $MODEL_SIZE" >&2; exit 1 ;;
esac
export MODEL_SIZE
export RUN_NAME="${RUN_NAME:-hybrid-infra-500-core-vllm-$MODEL_SIZE-20261027}"
export MODEL_PATH="/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/$MODEL_SIZE/midtrain/step0038147"
export OUTPUT_DIR="${OUTPUT_DIR:-/weka/oe-adapt-default/michaeln/open-instruct-hybrid/runs/$RUN_NAME}"
export CLUSTER="${CLUSTER:-ai2/rhea}"
# 500 optimizer updates, with 4 prompts x 4 completions per update.
export TOTAL_EPISODES=8000
exec bash scripts/train/olmo-hybrid-small/gsm8k_release_275m.sh "$BEAKER_IMAGE" \
    --max_result_age_steps 1 "$@"
