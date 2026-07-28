#!/bin/bash
# Launches one Beaker job that evaluates ALL checkpoints of a single DeepCoder-1.5B sweep run
# (--eval_only, one vLLM spin-up, looping internally over checkpoints -- see
# deepcoder_1_5b_eval_checkpoints_inner.sh) instead of one job per checkpoint.
#
# Required env vars:
#   EXP              - short name for this run lineage, e.g. baseline_n8_k16_seed2
#   CHECKPOINTS      - space-separated "path:step" pairs, e.g. "/path/step_100:100 /path/step_200:200"
set -e

: "${EXP:?Must set EXP, e.g. EXP=baseline_n8_k16_seed2}"
: "${CHECKPOINTS:?Must set CHECKPOINTS, space-separated path:step pairs}"

EXP_NAME="eval_deepcoder_1_5b_${EXP}"
WANDB_GROUP_NAME="eval_${EXP}"

NUM_GPUS="${NUM_GPUS:-4}"
BEAKER_IMAGE="${BEAKER_IMAGE:-nathanl/open_instruct_auto}"

CLUSTER="${CLUSTER:-ai2/jupiter}"
PRIORITY="${PRIORITY:-urgent}"
WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}"
NODES="${NODES:-1}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${EXP_NAME}_all_checkpoints" \
    --cluster ${CLUSTER} \
    --workspace ${WORKSPACE} \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --no_auto_dataset_cache \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes ${NODES} \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --env CHECKPOINTS="${CHECKPOINTS}" \
    --env WANDB_GROUP_NAME="${WANDB_GROUP_NAME}" \
    --env EXP_NAME="${EXP_NAME}" \
    --gpus $NUM_GPUS \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& source configs/beaker_configs/code_api_setup.sh \
\&\& bash scripts/train/qwen/deepcoder_1_5b_eval_checkpoints_inner.sh
