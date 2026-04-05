#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen25_05b_base_create_pass_dataset}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen_instruct_user_boxed_math}"
SOURCE_DATASET="${SOURCE_DATASET:-ai2-adapt-dev/rlvr_gsm8k_zs}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TOP_P="${TOP_P:-1.0}"
NUM_ENGINES="${NUM_ENGINES:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PASS_DATASET_REPO="${PASS_DATASET_REPO:-mnoukhov/gsm8k-platinum-qwen2.5-0.5b-base-1024samples-userprompt-topp1.0}"
BUCKET_DATASET_REPO="${BUCKET_DATASET_REPO:-mnoukhov/gsm8k-qwen2.5-0.5b-base-buckets}"
NUM_PER_BUCKET="${NUM_PER_BUCKET:-8}"
PASS_K="${PASS_K:-8}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/neptune \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- \
uv run scripts/data/rlvr/gsm8k_pass_at_32_dataset.py \
  --dataset ${SOURCE_DATASET} \
  --split ${SOURCE_SPLIT} \
  --model ${MODEL_NAME} \
  --chat-template ${CHAT_TEMPLATE} \
  --num-samples ${NUM_SAMPLES} \
  --max-tokens ${MAX_TOKENS} \
  --top-p ${TOP_P} \
  --num_engines ${NUM_ENGINES} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --push-to-hub ${PASS_DATASET_REPO} \
  --save-local-dir /weka/oe-adapt-default/allennlp/deletable_rollouts/michaeln/ \
\&\& uv run scripts/data/rlvr/create_gsm8k_pass_rate_buckets.py \
  --input-dataset ${PASS_DATASET_REPO} \
  --split test \
  --num-per-bucket ${NUM_PER_BUCKET} \
  --k ${PASS_K} \
  --buckets 0% 5% 10% 25% \
  --output-dataset ${BUCKET_DATASET_REPO} \
  --push-layout all
