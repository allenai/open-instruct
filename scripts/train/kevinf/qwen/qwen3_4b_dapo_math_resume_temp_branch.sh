#!/bin/bash
set -euo pipefail

if [[ -z "${BASE_CHECKPOINT_STATE_DIR:-}" ]]; then
    echo "BASE_CHECKPOINT_STATE_DIR must point to the source checkpoint-state directory."
    exit 1
fi

if [[ -z "${TEMPERATURE:-}" ]]; then
    echo "TEMPERATURE must be set for this continuation branch."
    exit 1
fi

RESUME_STEP="${RESUME_STEP:-500}"
BASE_STEP_TAG="${BASE_STEP_TAG:-global_step${RESUME_STEP}}"
EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_temp_fork}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_t${TEMPERATURE}_from_step${RESUME_STEP}_$(date +%Y%m%d_%H%M%S)}"
BRANCH_CHECKPOINT_STATE_DIR="${BRANCH_CHECKPOINT_STATE_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/${RUN_NAME}}"

NUM_GPUS="${NUM_GPUS:-8}"
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
shift || true

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
BUDGET="${BUDGET:-ai2/oe-omai}"

EXTRA_ARGS=""
for arg in "$@"; do
    printf -v quoted_arg "%q" "$arg"
    EXTRA_ARGS+=" ${quoted_arg}"
done

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env BASE_CHECKPOINT_STATE_DIR="${BASE_CHECKPOINT_STATE_DIR}" \
    --env BASE_STEP_TAG="${BASE_STEP_TAG}" \
    --env BRANCH_CHECKPOINT_STATE_DIR="${BRANCH_CHECKPOINT_STATE_DIR}" \
    --env EXP_NAME="${EXP_NAME}" \
    --env RUN_NAME="${RUN_NAME}" \
    --env TEMPERATURE="${TEMPERATURE}" \
    --gpus $NUM_GPUS \
    --budget ${BUDGET} \
    -- \
bash scripts/train/kevinf/qwen/qwen3_4b_dapo_math_resume_temp_branch_remote.sh ${EXTRA_ARGS}
