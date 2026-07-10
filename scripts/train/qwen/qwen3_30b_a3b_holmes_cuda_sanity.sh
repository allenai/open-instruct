#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-holmes-cuda-sanity}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"

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
    -- python scripts/train/qwen/qwen3_30b_a3b_holmes_cuda_sanity.py
