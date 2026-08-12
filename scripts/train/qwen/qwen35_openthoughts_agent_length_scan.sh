#!/bin/bash
set -euo pipefail

DATASET_SIZE="${DATASET_SIZE:-1K}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
DATASET="open-thoughts/OpenThoughts-Agent-SFT-${DATASET_SIZE}"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/OpenThoughts-Agent-SFT-${DATASET_SIZE}/qwen3.5-35b-a3b-olmo_thinker"
OUTPUT_PATH="${OUTPUT_DIR}/token-length-statistics.json"

uv run python mason.py \
    --task_name "qwen35-openthoughts-agent-${DATASET_SIZE,,}-length-scan" \
    --description "Qwen 3.5 token-length scan for ${DATASET}" \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --timeout 12h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --no_auto_dataset_cache \
    --gpus 0 \
    --env PYTHONPATH="${PROJECT_ROOT}/open-instruct" \
    -- python "${PROJECT_ROOT}/open-instruct/scripts/data/get_sft_token_length_stats.py" \
        --dataset "$DATASET" \
        --tokenizer Qwen/Qwen3.5-35B-A3B \
        --chat-template olmo_thinker \
        --output "$OUTPUT_PATH"
