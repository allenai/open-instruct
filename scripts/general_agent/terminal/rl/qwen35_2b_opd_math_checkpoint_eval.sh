#!/bin/bash

# Post-hoc AIME 2025 + BRUMO 2025 pass@1/pass@8 sweep for the Qwen3.5-2B
# math OPD run. Each checkpoint is evaluated independently on one H100.
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
shift

CHECKPOINT_ROOT="/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/qwen35_2b_opd_from_qwen35_9b_math_4node__42__1788510641"
PRIORITY="${PRIORITY:-urgent}"
MODEL_FILTER="${MODEL_FILTER:-.*}"
EVAL_CODE_DATASET="${EVAL_CODE_DATASET:-}"
EVAL_RUNNER="scripts/eval/run_math_vllm.sh"
EVAL_VLLM_WHEEL_URL="${EVAL_VLLM_WHEEL_URL:-https://wheels.vllm.ai/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}"
MASON_DATASET_ARGS=()
if [[ -n "$EVAL_CODE_DATASET" ]]; then
    EVAL_RUNNER="/eval/run_math_vllm.sh"
    MASON_DATASET_ARGS=(--beaker_datasets "/eval:$EVAL_CODE_DATASET")
fi

MODEL_LABELS=(base step_1 step_20 step_40 step_60 step_80 step_99)
MODEL_PATHS=(
    Qwen/Qwen3.5-2B
    "${CHECKPOINT_ROOT}_checkpoints/step_1"
    "${CHECKPOINT_ROOT}_checkpoints/step_20"
    "${CHECKPOINT_ROOT}_checkpoints/step_40"
    "${CHECKPOINT_ROOT}_checkpoints/step_60"
    "${CHECKPOINT_ROOT}_checkpoints/step_80"
    "$CHECKPOINT_ROOT"
)

for index in "${!MODEL_LABELS[@]}"; do
    label="${MODEL_LABELS[$index]}"
    model="${MODEL_PATHS[$index]}"
    if [[ ! "$label" =~ $MODEL_FILTER ]]; then
        continue
    fi
    WEIGHT_ARGS=()
    if [[ "$label" != "base" ]]; then
        WEIGHT_ARGS=(
            --strip-weight-prefix model.language_model.
            --weight-prefix-replacement model.
        )
    fi
    uv run python mason.py \
        --task_name "qwen35-2b-opd-math-eval-${label//_/-}" \
        --description "Qwen3.5-2B math OPD checkpoint sweep: ${label}, AIME 2025 + BRUMO 2025, pass@1/pass@8" \
        --cluster ai2/jupiter \
        --workspace ai2/open-instruct-dev \
        --priority "$PRIORITY" \
        --pure_docker_mode \
        --image "$BEAKER_IMAGE" \
        --min_runtime 2h \
        --auto_resume \
        --num_nodes 1 \
        --max_retries 2 \
        --timeout 8h \
        --gpus 1 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env VLLM_DISABLE_COMPILE_CACHE=1 \
        --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        --env EVAL_VLLM_WHEEL_URL="$EVAL_VLLM_WHEEL_URL" \
        "${MASON_DATASET_ARGS[@]}" \
        --no_auto_dataset_cache \
        -- \
    bash "$EVAL_RUNNER" \
        --model "$model" \
        --model-label "$label" \
        --datasets \
            mnoukhov/aime_2025_openinstruct \
            mnoukhov/brumo_2025_openinstruct \
        --split train \
        --chat-template qwen_instruct_user_boxed_math \
        --samples-per-prompt 8 \
        --temperature 1.0 \
        --top-p 1.0 \
        --max-prompt-tokens 2048 \
        --max-response-tokens 16384 \
        "${WEIGHT_ARGS[@]}" \
        --seed 42 \
        --output-dir /output "$@"
done
