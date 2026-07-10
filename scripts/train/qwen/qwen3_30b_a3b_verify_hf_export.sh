#!/bin/bash
set -euo pipefail

BEAKER_IMAGE="${1:?Pass the Open Instruct Beaker image as the first argument}"
PROJECT_ROOT="${PROJECT_ROOT:-/weka/oe-adapt-default/jacobm/olmoe3/post-training}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355/step247}"
HF_MODEL_PATH="${HF_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/qwen3-30b-a3b-dolci-think-olmo-core-sft-100k-20260710-174355-hf}"
HF_CACHE="${HF_CACHE:-${PROJECT_ROOT}/checkpoints/.hf-cache}"
OUTPUT_JSON="${OUTPUT_JSON:-${HF_MODEL_PATH}/logit-verification.json}"
DTYPE="${DTYPE:-bfloat16}"
RTOL="${RTOL:-0.02}"
ATOL="${ATOL:-0.02}"
LAYERWISE="${LAYERWISE:-true}"
SKIP_ASSERT_CLOSE="${SKIP_ASSERT_CLOSE:-false}"
MAX_MEAN_ABS_DIFF="${MAX_MEAN_ABS_DIFF:-}"
MIN_COSINE_SIMILARITY="${MIN_COSINE_SIMILARITY:-}"
MIN_TOP1_AGREEMENT="${MIN_TOP1_AGREEMENT:-}"
EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-sft-100k-verify-hf-export}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"

layerwise_args=()
if [[ "$LAYERWISE" == "true" ]]; then
    layerwise_args+=(--layerwise)
fi

acceptance_args=()
if [[ "$SKIP_ASSERT_CLOSE" == "true" ]]; then
    acceptance_args+=(--skip-assert-close)
fi
if [[ -n "$MAX_MEAN_ABS_DIFF" ]]; then
    acceptance_args+=(--max-mean-abs-diff "$MAX_MEAN_ABS_DIFF")
fi
if [[ -n "$MIN_COSINE_SIMILARITY" ]]; then
    acceptance_args+=(--min-cosine-similarity "$MIN_COSINE_SIMILARITY")
fi
if [[ -n "$MIN_TOP1_AGREEMENT" ]]; then
    acceptance_args+=(--min-top1-agreement "$MIN_TOP1_AGREEMENT")
fi

uv run python mason.py \
    --task_name "$EXP_NAME" \
    --description "$RUN_NAME" \
    --cluster ai2/holmes \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --timeout 3h \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --non_resumable \
    --no_auto_dataset_cache \
    --num_nodes 1 \
    --gpus 1 \
    --env HF_HOME="$HF_CACHE" \
    --env HF_HUB_CACHE="$HF_CACHE" \
    -- python "${PROJECT_ROOT}/OLMo-core/src/scripts/verify_qwen3_moe_hf_to_olmo_logits.py" \
    --hf-model "$HF_MODEL_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --hf-device cuda \
    --device cuda \
    --dtype "$DTYPE" \
    --rtol "$RTOL" \
    --atol "$ATOL" \
    "${layerwise_args[@]}" \
    "${acceptance_args[@]}" \
    --output-json "$OUTPUT_JSON"
