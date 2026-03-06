#!/bin/bash

set -euo pipefail

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/qwen3_4b_base_dapo_20260304_165842_checkpoints}"
STEP_START="${STEP_START:-100}"
STEP_END="${STEP_END:-2000}"
STEP_INCREMENT="${STEP_INCREMENT:-100}"

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_20260304_165842_eval_temp1.0_topp0.95}"
RUN_NAME="${RUN_NAME:-qwen3_4b_base_dapo_20260304_165842_eval_temp1.0_topp0.95}"

# Keep uv cache writable in restricted environments.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

for step in $(seq "${STEP_START}" "${STEP_INCREMENT}" "${STEP_END}"); do
    checkpoint_path="${CHECKPOINT_ROOT}/step_${step}"
    if [[ ! -d "${checkpoint_path}" ]]; then
        echo "Skipping missing checkpoint: ${checkpoint_path}"
        continue
    fi

    echo "Launching eval for ${checkpoint_path}"
    bash scripts/train/qwen/qwen3_4b_dapo_eval.sh \
        --model_name_or_path "${checkpoint_path}" \
        --eval_top_p 0.95 \
        --eval_temperature 1.0 \
        --exp_name "${EXP_NAME}" \
        --run_name "${RUN_NAME}" \
        --eval_only_set_checkpoint "${step}"
done
