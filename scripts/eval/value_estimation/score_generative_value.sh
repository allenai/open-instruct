#!/bin/bash
# Score with a generative value model served via vLLM.
set -euo pipefail

VALUE_MODEL_PATH="${1:?Value model path required}"
INPUT_DATASET_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"
OUTPUT_PATH="${3:-./value_estimation_data/generative_value_scores.parquet}"
CONDITIONING="${4:-none}"  # none | gt | correct_demo | rollout_context

uv run python -m open_instruct.value_estimation score_dataset \
    --input_dataset_path "${INPUT_DATASET_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --value_model_path "${VALUE_MODEL_PATH}" \
    --value_model_type generative \
    --gen_value_conditioning "${CONDITIONING}" \
    --gen_value_score_min 0 \
    --gen_value_score_max 10 \
    --gen_value_max_new_tokens 8 \
    --run_name "generative_value_${CONDITIONING}"
