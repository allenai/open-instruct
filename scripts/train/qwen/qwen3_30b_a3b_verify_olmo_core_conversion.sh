#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-verify-olmo-core-conversion}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-base-olmo}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"
VERIFY_SCRIPT="${PROJECT_ROOT}/OLMo-core/src/scripts/verify_qwen3_moe_hf_to_olmo_logits.py"

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes 1 \
    --gpus 1 \
    --env HF_HOME="$HF_CACHE" \
    --env HF_HUB_CACHE="$HF_CACHE" \
    -- python "$VERIFY_SCRIPT" \
    --hf-model Qwen/Qwen3-30B-A3B-Base \
    --checkpoint-path "$MODEL_PATH" \
    --hf-device cpu \
    --device cuda \
    --dtype bfloat16 \
    --rtol 0.02 \
    --atol 0.02 \
    --output-json "$MODEL_PATH/logit-verification.json"
