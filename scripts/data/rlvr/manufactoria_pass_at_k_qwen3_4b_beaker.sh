#!/usr/bin/env bash
# Launch Manufactoria pass@k dataset generation on Beaker for one dataset repo
# with train/test splits. The child script runs both splits sequentially.

set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
git_branch=$(git rev-parse --abbrev-ref HEAD)
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
IMAGE_NAME=open-instruct-integration-test-${sanitized_branch}
BEAKER_IMAGE="${BEAKER_IMAGE:-${BEAKER_USER}/${IMAGE_NAME}}"

NUM_GPUS="${NUM_GPUS:-8}"

EXP_NAME="${EXP_NAME:-manufactoria_pass128_qwen3_4b_step650}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/weka/oe-adapt-default/allennlp/deletable_rollouts/${BEAKER_USER}/manufactoria_pass_at_k}"

MASON_ENV=()
[[ -n "${DATASET_NAME+x}" ]] && MASON_ENV+=(--env "DATASET_NAME=${DATASET_NAME}")
[[ -n "${PUSH_DATASET_REPO+x}" ]] && MASON_ENV+=(--env "PUSH_DATASET_REPO=${PUSH_DATASET_REPO}")
[[ -n "${MODEL_NAME+x}" ]] && MASON_ENV+=(--env "MODEL_NAME=${MODEL_NAME}")
[[ -n "${NUM_SAMPLES+x}" ]] && MASON_ENV+=(--env "NUM_SAMPLES=${NUM_SAMPLES}")
[[ -n "${TRAIN_SPLIT+x}" ]] && MASON_ENV+=(--env "TRAIN_SPLIT=${TRAIN_SPLIT}")
[[ -n "${TEST_SPLIT+x}" ]] && MASON_ENV+=(--env "TEST_SPLIT=${TEST_SPLIT}")
[[ -n "${NUM_GPUS+x}" ]] && MASON_ENV+=(--env "NUM_ENGINES=${NUM_GPUS}")

uv run python mason.py \
  --cluster ai2/neptune ai2/saturn ai2/rhea \
  --workspace ai2/oe-adapt-code \
  --priority high \
  --preemptible \
  --pure_docker_mode \
  --budget ai2/oe-adapt \
  --description "${EXP_NAME}" \
  --image "${BEAKER_IMAGE}" \
  --num_nodes 1 \
  --gpus ${NUM_GPUS} \
  --max_retries 0 \
  --no_auto_dataset_cache \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env "SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR}" \
  "${MASON_ENV[@]}" \
  -- \
  source configs/beaker_configs/ray_node_setup.sh \&\& \
  source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
  bash scripts/data/rlvr/manufactoria_pass_at_k_qwen3_4b.sh
