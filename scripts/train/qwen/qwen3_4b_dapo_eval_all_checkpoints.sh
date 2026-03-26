#!/bin/bash

set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-hbx/JustRL-DeepSeek-1.5B}"
STEP_START="${STEP_START:-200}"
STEP_END="${STEP_END:-4200}"
STEP_INCREMENT="${STEP_INCREMENT:-200}"

EXP_NAME="${EXP_NAME:-deepseek1.5b_aime_brumo_eval}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-${EXP_NAME}}"

# Keep uv cache writable in restricted environments.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

for step in $(seq "${STEP_START}" "${STEP_INCREMENT}" "${STEP_END}"); do
    revision=$(printf "step_%04d" "${step}")
    echo "Launching eval for ${HF_REPO_ID} revision ${revision}"

        # --model_revision "${revision}" \
        #
    NUM_GPUS=4 EXP_NAME=$EXP_NAME bash scripts/train/qwen/qwen3_4b_dapo_math.sh \
        --model_name_or_path "${HF_REPO_ID}" \
        --exp_name "${EXP_NAME}" \
        --run_name "${RUN_NAME}_step_${step}" \
        --wandb_group_name "${WANDB_GROUP_NAME}" \
        --eval_only \
        --eval_only_set_checkpoint "${step}" \
        --eval_response_length 32000 \
        --eval_temperature 0.7 \
        --eval_top_p 0.9 \
        "$@"
done
