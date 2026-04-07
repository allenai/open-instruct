#!/bin/bash
set -euo pipefail

USER="${USER:-michaeln}"
EXP_NAME="${EXP_NAME:-manufactoria_create_pass32_dataset}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
INPUT_DATASET="${INPUT_DATASET:-manufactoria/basic_mix_test}"
INPUT_SPLIT="${INPUT_SPLIT:-train}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-tulu}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
NUM_ENGINES="${NUM_ENGINES:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/weka/oe-adapt-default/allennlp/deletable_rollouts/${USER}/manufactoria_pass_datasets}"

: "${PASS_DATASET_REPO:?Set PASS_DATASET_REPO to the HF repo for pass@32 outputs}"

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --cluster ai2/jupiter \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --gpus "${NUM_GPUS}" \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    -- \
uv run scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \
  --dataset "${INPUT_DATASET}" \
  --split "${INPUT_SPLIT}" \
  --model "${MODEL_NAME_OR_PATH}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-tokens "${MAX_TOKENS}" \
  --top-p 1.0 \
  --num-engines "${NUM_ENGINES}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --push-to-hub "${PASS_DATASET_REPO}" \
  --save-local-dir "${SAVE_LOCAL_DIR}"
