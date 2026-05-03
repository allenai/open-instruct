#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3_4b_dapo_math_create_pass_dataset}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
NUM_GPUS="${NUM_GPUS:-8}"
CLUSTER="${CLUSTER:-ai2/jupiter ai2/ceres}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus ${NUM_GPUS} \
    --budget ai2/oe-adapt \
    -- \
uv run scripts/data/rlvr/aime_pass_at_k_dataset.py \
    --dataset hamishivi/DAPO-Math-17k-Processed_filtered \
    --splits train \
    --model Qwen/Qwen3-4B-Base \
    --chat-template qwen_instruct_user_boxed_math \
    --num-samples 32 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-tokens 8192 \
    --num-engines 8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --push-to-hub mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
    --save-local-dir /weka/oe-adapt-default/allennlp/deletable_rollouts/michaeln/dapo_math_pass_at_k \
    "$@" \
\&\& uv run scripts/data/rlvr/create_gsm8k_pass_rate_quartiles.py \
    --input-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples \
    --split train \
    --output-dataset mnoukhov/dapo-math-17k-processed-filtered-qwen3-4b-base-32samples-quartiles \
    --dataset-name-prefix math_dapo
