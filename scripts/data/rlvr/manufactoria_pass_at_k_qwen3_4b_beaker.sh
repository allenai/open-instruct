#!/usr/bin/env bash
# Launch manufactoria pass@32 dataset generation on Beaker (8 GPUs), matching
# scripts/train/manufactoria/qwen3_4b_phase1_has_8gpu.sh cluster/image style.
# Output datasets include a per-row `difficulty` list (quartiles per test) instead of separate bucket subsets.
#
# Hub: child script defaults to mnoukhov/manufactoria-has-{train,test}-qwen3-4b-instruct-pass32
# (set HF_DATASETS_OWNER to change the namespace). Override full names with PUSH_TRAIN_REPO / PUSH_TEST_REPO.
# Use PUSH_TRAIN_REPO= to skip pushing train (empty value, in the shell you launch mason from).

set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
git_branch=$(git rev-parse --abbrev-ref HEAD)
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
IMAGE_NAME=open-instruct-integration-test-${sanitized_branch}
BEAKER_IMAGE="${BEAKER_IMAGE:-${BEAKER_USER}/${IMAGE_NAME}}"

EXP_NAME="${EXP_NAME:-manufactoria_pass32_qwen3_4b}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/weka/oe-adapt-default/allennlp/deletable_rollouts/${BEAKER_USER}/manufactoria_pass_at_k}"

# Only forward PUSH_* when set on the submit host; if unset, the child script uses default Hub names.
MASON_PUSH_ENV=()
[[ -n "${PUSH_TRAIN_REPO+x}" ]] && MASON_PUSH_ENV+=(--env "PUSH_TRAIN_REPO=${PUSH_TRAIN_REPO}")
[[ -n "${PUSH_TEST_REPO+x}" ]] && MASON_PUSH_ENV+=(--env "PUSH_TEST_REPO=${PUSH_TEST_REPO}")
[[ -n "${HF_DATASETS_OWNER+x}" ]] && MASON_PUSH_ENV+=(--env "HF_DATASETS_OWNER=${HF_DATASETS_OWNER}")

uv run python mason.py \
  --cluster ai2/jupiter \
  --cluster ai2/saturn \
  --cluster ai2/ceres \
  --workspace ai2/oe-adapt-code \
  --priority high \
  --preemptible \
  --pure_docker_mode \
  --budget ai2/oe-adapt \
  --description "${EXP_NAME}" \
  --image "${BEAKER_IMAGE}" \
  --num_nodes 1 \
  --gpus 8 \
  --max_retries 0 \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env "SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR}" \
  "${MASON_PUSH_ENV[@]}" \
  -- \
  source configs/beaker_configs/ray_node_setup.sh \&\& \
  source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
  bash scripts/data/rlvr/manufactoria_pass_at_k_qwen3_4b.sh
