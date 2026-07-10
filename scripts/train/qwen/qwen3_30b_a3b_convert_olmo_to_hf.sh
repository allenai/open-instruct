#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355/step247}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355-hf}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"
EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-sft-100k-convert-to-hf}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
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
    --env HF_HOME="$HF_CACHE" \
    --env HF_HUB_CACHE="$HF_CACHE" \
    -- python "${PROJECT_ROOT}/OLMo-core/src/scripts/convert_qwen3_moe_olmo_to_hf.py" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --hf-model-name Qwen/Qwen3-30B-A3B-Base \
    --tokenizer-name Qwen/Qwen3-30B-A3B \
    --dtype bfloat16 \
    --max-shard-size 5GB
