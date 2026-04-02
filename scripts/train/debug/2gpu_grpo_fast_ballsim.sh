#!/bin/bash

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

export REPO_PATH=$(pwd)
export BEAKER_LEADER_REPLICA_HOSTNAME=$(hostname)

source .venv/bin/activate
# source configs/beaker_configs/ray_node_setup.sh
source configs/beaker_configs/ballsim_api_setup.sh
python open_instruct/grpo_fast.py \
    --dataset_mixer_list bouncingsim/bouncingsim-MULTIOBJ-basic 32 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list bouncingsim/bouncingsim-MULTIOBJ-basic 16 \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length 4096 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --learning_rate 3e-7 \
    --total_episodes 64 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 4 \
    --gradient_checkpointing \
    --push_to_hub false \
    --system_prompt_override_file scripts/train/debug/ballsim_system_prompt.txt \
    --ballsim_api_url "${BALLSIM_API_URL}" \
    --ballsim_max_execution_time 1.0 \
    --ballsim_scoring_mode all_pass
