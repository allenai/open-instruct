#!/bin/bash

set -euo pipefail

STEP_START=200
STEP_END=2800
STEP_INCREMENT=200

TASK_NAME="${TASK_NAME:-olmo31_7b_rlzero_math_eval_all_checkpoints}"
RUN_NAME_PREFIX="olmo31_7b_rlzero_math_eval_all_checkpoints"

for step in $(seq "${STEP_START}" "${STEP_INCREMENT}" "${STEP_END}"); do
    revision=$(printf "step_%04d" "${step}")
    run_name="${RUN_NAME_PREFIX}_${revision}"

    echo "Launching eval for allenai/Olmo-3.1-7B-RL-Zero-Math revision ${revision}"
    TASK_NAME="${TASK_NAME}" \
    DESCRIPTION="${run_name}" \
    bash scripts/train/olmo3/7b_rlzero_math_eval.sh \
        --model_name_or_path "allenai/Olmo-3.1-7B-RL-Zero-Math" \
        --model_revision "${revision}" \
        --exp_name "olmo31_7b_rlzero_math_eval_all_checkpoints" \
        --run_name "${run_name}" \
        --eval_only_set_checkpoint "${step}" \
        "$@"
done
