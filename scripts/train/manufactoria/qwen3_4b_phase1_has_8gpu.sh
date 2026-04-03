#!/bin/bash

set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)
# Sanitize the branch name to remove invalid characters for Beaker names
# Beaker names can only contain letters, numbers, -_. and may not start with -
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
IMAGE_NAME=open-instruct-integration-test-${sanitized_branch}

BEAKER_IMAGE="${1:-${BEAKER_USER}/${IMAGE_NAME}}"

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_32k}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

CLIP_HIGH=0.3
LR=5e-7
SCORE_MODE=pass_rate
TRAIN_LIST="manufactoria/has_train 1.0"
EVAL_LIST="manufactoria/has_test 50"
DESCRIPTION="manufactoria_has_phase1_pass_rate_8gpu"
EXP_NAME="Qwen3_4B_Instruct_manufactoria_has_phase1_pass_rate"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --preemptible \
    --pure_docker_mode \
    --budget ai2/oe-adapt \
    --description "${DESCRIPTION}" \
    --image "${BEAKER_IMAGE}" \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --timeout 8h \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& \
    source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
    python open_instruct/grpo_fast.py \
    --exp_name "${EXP_NAME}" \
    --beta 0.0 \
    --load_ref_policy false \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --kl_estimator 3 \
    --learning_rate "${LR}" \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ${TRAIN_LIST} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ${EVAL_LIST} \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path "${BASE_MODEL}" \
    --apply_verifiable_reward true \
    --manufactoria_api_url \$MANUFACTORIA_API_URL/test_solution \
    --manufactoria_scoring_mode "${SCORE_MODE}" \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --total_episodes 1000000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher "${CLIP_HIGH}" \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false "$@"
