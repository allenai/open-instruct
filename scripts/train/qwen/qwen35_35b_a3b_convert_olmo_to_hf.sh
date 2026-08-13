#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Qwen 3.5 Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an OLMo-core step directory}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_PATH}-hf}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3.5-35b-a3b-olmo_thinker/smoke-1k-4096-cpu/tokenizer}"
VERIFY_ONLY="${VERIFY_ONLY:-false}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-65536}"

convert_args=()
if [[ "$VERIFY_ONLY" == "true" ]]; then
    convert_args+=(--verify-only)
fi

uv run python mason.py \
    --task_name qwen35-35b-a3b-convert-olmo-to-hf \
    --description "Export ${CHECKPOINT_PATH} to native Qwen 3.5 HF format" \
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
    -- python "${PROJECT_ROOT}/OLMo-core/src/scripts/convert_qwen3_moe_olmo_to_hf.py" \
        --checkpoint-path "$CHECKPOINT_PATH" \
        --output-path "$OUTPUT_PATH" \
        --hf-model-name Qwen/Qwen3.5-35B-A3B-Base \
        --tokenizer-name "$TOKENIZER_NAME" \
        --generation-config-name Qwen/Qwen3.5-35B-A3B \
        --dtype bfloat16 \
        --max-shard-size 5GB \
        --max-position-embeddings "$MAX_POSITION_EMBEDDINGS" \
        "${convert_args[@]}"
