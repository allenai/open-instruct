#!/bin/bash
set -euo pipefail

DATASET_SIZE="${DATASET_SIZE:-1K}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
NUM_GPUS="${NUM_GPUS:-1}"
BEAKER_CLUSTER="${BEAKER_CLUSTER:-ai2/jupiter}"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/olmo-instruct}"
BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
DATASET="open-thoughts/OpenThoughts-Agent-SFT-${DATASET_SIZE}"
OUTPUT_DIR="${PROJECT_ROOT}/datasets/OpenThoughts-Agent-SFT-${DATASET_SIZE}/qwen3.5-35b-a3b-olmo_thinker-${MAX_SEQ_LENGTH}"

uv run python mason.py \
    --task_name "qwen35-openthoughts-agent-${DATASET_SIZE,,}-tokenize-${MAX_SEQ_LENGTH}" \
    --description "Qwen 3.5 tokenization for ${DATASET} at ${MAX_SEQ_LENGTH} tokens" \
    --cluster "$BEAKER_CLUSTER" \
    --workspace "$BEAKER_WORKSPACE" \
    --priority urgent \
    --max_retries 3 \
    --timeout 24h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --preemptible \
    --no_auto_dataset_cache \
    --gpus "$NUM_GPUS" \
    --env PYTHONPATH="${PROJECT_ROOT}/open-instruct" \
    -- python "${PROJECT_ROOT}/open-instruct/scripts/data/convert_sft_data_for_olmocore.py" \
        --dataset_mixer_list "$DATASET" 1.0 \
        --tokenizer_name_or_path Qwen/Qwen3.5-35B-A3B \
        --output_dir "$OUTPUT_DIR" \
        --visualize True \
        --chat_template_name olmo_thinker \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --ensure_terminal_eos_after_truncation True \
        --resume True
