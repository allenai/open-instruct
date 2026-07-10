#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:-jacobm/open-instruct-integration-test-jacobm-olmoe3-post-training}"

PROJECT_ROOT=/weka/oe-adapt-default/jacobm/olmoe3/post-training
OUTPUT_DIR=${PROJECT_ROOT}/datasets/Dolci-Think-SFT-32B/qwen3-30b-a3b-olmo_thinker/100k

uv run python mason.py \
    --task_name qwen3-30b-a3b-dolci-think-tokenize-100k \
    --description "Qwen3 30B A3B Dolci-Think tokenization (100k)" \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 3 \
    --timeout 6h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --no-host-networking \
    --preemptible \
    --no_auto_dataset_cache \
    --gpus 8 \
    -- python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list allenai/Dolci-Think-SFT-32B 100000 \
        --tokenizer_name_or_path Qwen/Qwen3-30B-A3B \
        --output_dir "$OUTPUT_DIR" \
        --visualize True \
        --chat_template_name olmo_thinker \
        --max_seq_length 32768 \
        --resume True
