#!/bin/bash

set -euo pipefail

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-allenai/Olmo-3.1-7B-RL-Zero-Math}"
STEP_START="${STEP_START:-200}"
STEP_END="${STEP_END:-2800}"
STEP_INCREMENT="${STEP_INCREMENT:-200}"

EXP_NAME="${EXP_NAME:-olmo31_7b_rlzero_math_eval_all_checkpoints}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${EXP_NAME}}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-${EXP_NAME}}"

for step in $(seq "${STEP_START}" "${STEP_INCREMENT}" "${STEP_END}"); do
    revision=$(printf "step_%04d" "${step}")
    run_name="${RUN_NAME_PREFIX}_${revision}"

    echo "Launching eval for ${MODEL_NAME_OR_PATH} revision ${revision}"
    WANDB_GROUP_NAME="${WANDB_GROUP_NAME}" \
    bash scripts/train/olmo3/7b_rlzero_math_eval.sh \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --model_revision "${revision}" \
        --exp_name "${EXP_NAME}" \
        --run_name "${run_name}" \
        --eval_only_set_checkpoint "${step}" \
        "$@"
done
