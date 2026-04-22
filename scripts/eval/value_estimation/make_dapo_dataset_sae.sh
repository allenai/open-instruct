#!/bin/bash
# Build the value-estimation dataset from DAPO math using SAE-based probe positions.
# Probes are placed at tokens with predicted probability < sae_threshold (0.2),
# downsampled to max_probes (16) evenly-spaced entries — matching training-time boundaries.
set -euo pipefail

MODEL_NAME_OR_PATH="${1:-Qwen/Qwen3-4B-Base}"
OUTPUT_PATH="${2:-./value_estimation_data/dapo_math_100pairs_sae.parquet}"

uv run python -m open_instruct.value_estimation make_dataset \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered \
    --target_num_pairs 100 \
    --rollouts_per_prompt 8 \
    --continuations_per_probe 32 \
    --probe_mode sae \
    --sae_threshold 0.2 \
    --max_probes 16 \
    --max_response_length 8192
