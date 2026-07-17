#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:-jacobm/open-instruct-integration-test-jacobm-olmoe3-post-training}"
DATASET_VARIANT="${DATASET_VARIANT:-100k}"

case "$DATASET_VARIANT" in
    10k)
        MIXER_AMOUNT=10000
        DEFAULT_TIMEOUT=6h
        ;;
    100k)
        MIXER_AMOUNT=100000
        DEFAULT_TIMEOUT=6h
        ;;
    full)
        MIXER_AMOUNT=1.0
        DEFAULT_TIMEOUT=24h
        ;;
    *)
        echo "DATASET_VARIANT must be one of: 10k, 100k, full" >&2
        exit 2
        ;;
esac

TIMEOUT="${TIMEOUT:-$DEFAULT_TIMEOUT}"

PROJECT_ROOT=/weka/oe-adapt-default/jacobm/olmoe3/post-training
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3-30b-a3b-olmo_thinker-terminal-eos-v2/${DATASET_VARIANT}}"

uv run python mason.py \
    --task_name "qwen3-30b-a3b-dolci-think-tokenize-${DATASET_VARIANT}-terminal-eos-v2" \
    --description "Qwen3 30B A3B Dolci-Think tokenization (${DATASET_VARIANT}, terminal EOS v2)" \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 3 \
    --timeout "$TIMEOUT" \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --preemptible \
    --no_auto_dataset_cache \
    --gpus 8 \
    --env PYTHONPATH="${PROJECT_ROOT}/open-instruct" \
    -- python "${PROJECT_ROOT}/open-instruct/scripts/data/convert_sft_data_for_olmocore.py" \
        --dataset_mixer_list allenai/Dolci-Think-SFT-32B "$MIXER_AMOUNT" \
        --tokenizer_name_or_path Qwen/Qwen3-30B-A3B \
        --output_dir "$OUTPUT_DIR" \
        --visualize True \
        --chat_template_name olmo_thinker \
        --max_seq_length 32768 \
        --ensure_terminal_eos_after_truncation True \
        --resume True
