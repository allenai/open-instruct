#!/bin/bash

# Post-hoc AIME 2025 + BRUMO 2025 pass@1/pass@8 sweep for the Qwen3.5-2B
# math OPD run. Each checkpoint is evaluated independently on one H100.
set -euo pipefail

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"
shift

CHECKPOINT_ROOT="/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/qwen35_2b_opd_from_qwen35_9b_math_4node__42__1788510641"
PRIORITY="${PRIORITY:-urgent}"
MODEL_FILTER="${MODEL_FILTER:-.*}"

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
        --env VLLM_USE_V1=1 \
        --no_auto_dataset_cache \
        -- \
    uv run python scripts/eval/math_vllm.py \
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
        --seed 42 \
        --output-dir /output "$@"
done
