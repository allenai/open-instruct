#!/bin/bash
# Score with an LM-yesno value model (Yes/No probability as the value).
set -euo pipefail

VALUE_MODEL_PATH="${1:?Value model path required}"
INPUT_DATASET_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"
OUTPUT_PATH="${3:-./value_estimation_data/lm_yesno_scores.parquet}"
TEMPLATE="${4:-lm_yesno}"  # lm_yesno | lm_yesno_blind | lm_yesno_siblings

uv run python -m open_instruct.value_estimation score_dataset \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --value_model_path "${VALUE_MODEL_PATH}" \
    --value_model_type lm_yesno \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template "${TEMPLATE}" \
    --run_name "lm_yesno_${TEMPLATE}"
