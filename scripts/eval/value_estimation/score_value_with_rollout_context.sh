#!/bin/bash
# Score a scalar value model with rollout-context conditioning (fellow sibling rollouts).
set -euo pipefail

VALUE_MODEL_PATH="${1:?Value model path required}"
INPUT_DATASET_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"
OUTPUT_PATH="${3:-./value_estimation_data/scalar_value_rollout_context_scores.parquet}"

uv run python -m open_instruct.value_estimation score_dataset \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --value_model_path "${VALUE_MODEL_PATH}" \
    --value_model_type scalar \
    --value_model_ground_truth_conditioning \
    --gt_conditioning_template rollout_context \
    --rollout_context_num_siblings 4 \
    --run_name scalar_value_rollout_context
