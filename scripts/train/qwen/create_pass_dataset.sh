#!/bin/bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen25_05b_create_pass_dataset}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"

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
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus 4 \
    --budget ai2/oe-adapt \
    -- \
uv run scripts/data/rlvr/gsm8k_pass_at_32_dataset.py \
  --dataset mnoukhov/gsm8k-platinum-openinstruct \
  --split test \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --chat-template qwen_instruct_boxed_math \
  --num-samples 1024 \
  --max-tokens 4096 \
  --tensor-parallel-size 4 \
  --push-to-hub mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples \
  --save-local-dir /weka/oe-adapt-default/allennlp/deletable_rollouts/michaeln/ \
\&\& uv run scripts/data/rlvr/create_gsm8k_pass_rate_buckets.py \
  --input-dataset mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples \
  --split test \
  --num-per-bucket 8 \
  --k 16 \
  --buckets 0% 5% 10% 25% \
  --output-dataset mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-buckets \
  --push-layout all
