#!/bin/bash
# Build the value-estimation dataset from DAPO math.
# Writes a parquet of (prompt, ground_truth, rollout, probe_positions, mc_values).
set -euo pipefail

MODEL_NAME_OR_PATH="${1:-Qwen/Qwen3-4B-Base}"
OUTPUT_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"
# Default matches training-time chat template for Base models. Pass "builtin" to use the
# model's own chat template (for Instruct models like Qwen3-4B-Instruct-2507).
CHAT_TEMPLATE_NAME="${3:-qwen_instruct_user_boxed_math}"

uv run python -m open_instruct.value_estimation make_dataset \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered \
    --target_num_pairs 100 \
    --rollouts_per_prompt 8 \
    --continuations_per_probe 32 \
    --probe_interval 1000 \
    --max_response_length 8192 \
    --chat_template_name "${CHAT_TEMPLATE_NAME}"
