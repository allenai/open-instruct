#!/bin/bash
set -euo pipefail

# Recover an HF checkpoint from the legacy `model_state_dict.pt` fallback
# produced by the old olmo-core save path.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/debug/convert_pt_state_dict_to_hf.sh \
#       /weka/oe-adapt-default/allennlp/deletable_checkpoint/qwen3_4b_base_dapo_20260501_153944/tmp-3m \
#       Qwen/Qwen3-4B-Base

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
SRC_DIR="${2:?src-dir required as 2nd arg}"
BASE_MODEL="${3:?base-model required as 3rd arg}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/ceres \
    --image "$BEAKER_IMAGE" \
    --description "Recover HF checkpoint from model_state_dict.pt at $SRC_DIR" \
    --pure_docker_mode \
    --no-host-networking \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --budget ai2/oe-adapt \
    --gpus 0 \
    --no_auto_dataset_cache \
    --task_name convert_pt_to_hf \
    -- uv run python scripts/train/convert_pt_state_dict_to_hf.py \
        --src-dir "$SRC_DIR" \
        --base-model "$BASE_MODEL"
