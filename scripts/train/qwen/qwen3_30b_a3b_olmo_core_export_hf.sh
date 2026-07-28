#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-olmo-core-export-hf}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an OLMo-core SFT checkpoint}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH for the exported Hugging Face checkpoint}"
HF_MODEL_NAME="${HF_MODEL_NAME:-Qwen/Qwen3-30B-A3B-Base}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-30B-A3B}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
CLUSTER="${CLUSTER:-ai2/holmes}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
PRIORITY="${PRIORITY:-urgent}"
TIMEOUT="${TIMEOUT:-6h}"
GPU_COUNT="${GPU_COUNT:-1}"

overwrite_args=()
if [[ "${SAVE_OVERWRITE:-false}" == "true" ]]; then
    overwrite_args+=(--save-overwrite)
fi

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" \
    --priority "$PRIORITY" \
    --timeout "$TIMEOUT" \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --no_auto_dataset_cache \
    --gpus "$GPU_COUNT" \
    --env TOKENIZERS_PARALLELISM=false \
    -- python -m olmo_core.nn.moe.v2.qwen_hf_export \
        --checkpoint-path "$CHECKPOINT_PATH" \
        --output-path "$OUTPUT_PATH" \
        --hf-model-name "$HF_MODEL_NAME" \
        --tokenizer-name "$TOKENIZER_NAME" \
        --dtype bfloat16 \
        --max-shard-size "$MAX_SHARD_SIZE" \
        --verify-after-export \
        "${overwrite_args[@]}"
