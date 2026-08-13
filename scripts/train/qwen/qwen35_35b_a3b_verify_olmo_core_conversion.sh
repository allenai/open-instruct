#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Qwen 3.5 Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/qwen3.5-35b-a3b-base-olmo}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"

uv run python mason.py \
    --task_name qwen35-35b-a3b-verify-olmo-conversion \
    --description "Compare Qwen 3.5 Base HF and converted OLMo-core logits" \
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
    -- python "${PROJECT_ROOT}/OLMo-core/src/scripts/verify_qwen3_moe_hf_to_olmo_logits.py" \
        --hf-model Qwen/Qwen3.5-35B-A3B-Base \
        --tokenizer-name Qwen/Qwen3.5-35B-A3B \
        --checkpoint-path "$MODEL_PATH" \
        --hf-device cuda \
        --device cuda \
        --dtype bfloat16 \
        --skip-assert-close \
        --max-mean-abs-diff 0.1 \
        --min-cosine-similarity 0.999 \
        --min-top1-agreement 0.9 \
        --output-json "$MODEL_PATH/logit-verification.json"
