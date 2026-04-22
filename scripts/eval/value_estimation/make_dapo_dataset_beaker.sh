#!/bin/bash
# Build the value-estimation dataset on Beaker using Qwen3-4B-Base.
# Results are written to /output/dapo_math_100pairs.parquet (Beaker result mount).
DDMM=$(date +"%d%m")
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -- python -m open_instruct.value_estimation make_dataset \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --output_path /output/dapo_math_100pairs.parquet \
    --dataset_name hamishivi/DAPO-Math-17k-Processed_filtered \
    --target_num_pairs 100 \
    --rollouts_per_prompt 8 \
    --continuations_per_probe 32 \
    --probe_interval 1000 \
    --max_response_length 8192
