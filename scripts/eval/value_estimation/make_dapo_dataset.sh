#!/bin/bash
# Build the value-estimation dataset from DAPO math.
# Writes a parquet of (prompt, ground_truth, rollout, probe_positions, mc_values).
set -euo pipefail

MODEL_NAME_OR_PATH="${1:-Qwen/Qwen3-4B-Base}"
OUTPUT_PATH="${2:-./value_estimation_data/dapo_math_100pairs.parquet}"

uv run python -m open_instruct.value_estimation make_dataset \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered \
    --target_num_pairs 100 \
    --rollouts_per_prompt 8 \
    --continuations_per_probe 32 \
    --probe_interval 1000 \
    --max_response_length 8192
