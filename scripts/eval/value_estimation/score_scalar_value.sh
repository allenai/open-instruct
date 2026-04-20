#!/bin/bash
# Score the value-estimation dataset with a scalar PPO value model (no conditioning).
set -euo pipefail

VALUE_MODEL_PATH="${1:?Value model path required}"
INPUT_DATASET_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"
OUTPUT_PATH="${3:-./value_estimation_data/scalar_value_scores.parquet}"

uv run python -m open_instruct.value_estimation score_dataset \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --value_model_path "${VALUE_MODEL_PATH}" \
    --value_model_type scalar \
    --run_name scalar_value_no_cond
