#!/bin/bash
set -euo pipefail

# Build a pass@32 + difficulty-quartile dataset from a 10k subset of DeepScaleR,
# sampled with Qwen3-4B-Base using the same chat template / sampling settings as
# scripts/train/qwen/qwen3_4b_deepscaler_math.sh.
#
# Launch with:
#   ./scripts/train/build_image_and_launch.sh scripts/train/qwen/create_deepscaler_pass_dataset.sh

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

EXP_NAME="${EXP_NAME:-qwen3_4b_deepscaler_create_pass_dataset}"
NUM_GPUS="${NUM_GPUS:-8}"
CLUSTER="${CLUSTER:-ai2/jupiter ai2/ceres}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"

RLVR_DATASET="${RLVR_DATASET:-mnoukhov/deepscaler-10k-rlvr}"
SAMPLES_DATASET="${SAMPLES_DATASET:-mnoukhov/deepscaler-10k-qwen3-4b-base-32samples}"
QUARTILES_DATASET="${QUARTILES_DATASET:-mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles}"

echo "Using Beaker image: $BEAKER_IMAGE"

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
uv run scripts/data/create_deepscaler_subset_rlvr.py \
    --push_to_hub \
    --repo_id ${RLVR_DATASET} \
    --num_examples 10000 \
    --seed 42 \
\&\& uv run scripts/data/rlvr/aime_pass_at_k_dataset.py \
    --dataset ${RLVR_DATASET} \
    --splits train \
    --model Qwen/Qwen3-4B-Base \
    --chat-template qwen_instruct_user_boxed_math \
    --num-samples 32 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-tokens 8192 \
    --num-engines ${NUM_GPUS} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --push-to-hub ${SAMPLES_DATASET} \
    --save-local-dir /weka/oe-adapt-default/allennlp/deletable_rollouts/${BEAKER_USER}/deepscaler_pass_at_k \
\&\& uv run scripts/data/rlvr/create_gsm8k_pass_rate_quartiles.py \
    --input-dataset ${SAMPLES_DATASET} \
    --split train \
    --output-dataset ${QUARTILES_DATASET} \
    --dataset-name-prefix math_deepscaler
