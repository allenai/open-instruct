#!/bin/bash
set -euo pipefail

USER=michaeln
EXP_NAME="${EXP_NAME:-olmo3_7b_create_aime24_25_pass_dataset}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-allenai/Olmo-3-1025-7B}"
INPUT_DATASET="${INPUT_DATASET:-allenai/aime2024-25-rlvr}"
INPUT_SPLITS="${INPUT_SPLITS:-test_2024 test_2025}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-olmo_thinker_rlzero}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_ENGINES="${NUM_ENGINES:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/weka/oe-adapt-default/allennlp/deletable_rollouts/${USER}/aime24_25_pass_datasets}"

: "${PASS_DATASET_REPO:?Set PASS_DATASET_REPO to the HF repo for raw pass@k outputs}"
: "${QUARTILES_DATASET_REPO:?Set QUARTILES_DATASET_REPO to the HF repo for quartile outputs}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/jupiter \
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
uv run scripts/data/rlvr/aime_pass_at_k_dataset.py \
  --dataset ${INPUT_DATASET} \
  --splits ${INPUT_SPLITS} \
  --model ${MODEL_NAME_OR_PATH} \
  --chat-template ${CHAT_TEMPLATE} \
  --num-samples ${NUM_SAMPLES} \
  --max-tokens 32768 \
  --top-p 1.0 \
  --num-engines ${NUM_ENGINES} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --push-to-hub ${PASS_DATASET_REPO} \
  --save-local-dir ${SAVE_LOCAL_DIR} \
\&\& uv run scripts/data/rlvr/create_aime_pass_rate_quartiles.py \
  --input-dataset ${PASS_DATASET_REPO} \
  --splits ${INPUT_SPLITS} \
  --output-dataset ${QUARTILES_DATASET_REPO}
