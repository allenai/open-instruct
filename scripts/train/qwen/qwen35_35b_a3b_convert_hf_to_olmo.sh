#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Qwen 3.5 Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B-Base}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3.5-35B-A3B}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/checkpoints/qwen3.5-35b-a3b-base-olmo}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"
DRY_RUN="${DRY_RUN:-false}"

convert_args=()
if [[ "$DRY_RUN" == "true" ]]; then
    convert_args+=(--dry-run)
fi

uv run python mason.py \
    --task_name "qwen35-35b-a3b-base-hf-to-olmo" \
    --description "Convert ${MODEL_NAME} to OLMo-core (dry_run=${DRY_RUN})" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --timeout 6h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes 1 \
    --gpus 1 \
    --env PYTHONPATH="${PROJECT_ROOT}/OLMo-core/src:${PROJECT_ROOT}/open-instruct" \
    --env HF_HOME="$HF_CACHE" \
    --env HF_HUB_CACHE="$HF_CACHE" \
    -- python "${PROJECT_ROOT}/OLMo-core/src/scripts/convert_qwen3_moe_hf_to_olmo.py" \
        --hf-model "$MODEL_NAME" \
        --tokenizer-name "$TOKENIZER_NAME" \
        --output-path "$OUTPUT_PATH" \
        --cache-dir "$HF_CACHE" \
        --attention-backend flash_4 \
        --device cuda \
        "${convert_args[@]}"
