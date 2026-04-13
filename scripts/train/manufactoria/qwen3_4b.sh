#!/bin/bash

set -euo pipefail

BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"

EXP="${EXP:-}"
EXP_NAME="${EXP_NAME:-qwen3_4b_it_manufac_${EXP}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${NUM_GPUS:-8}"

uv run python mason.py \
    --cluster ai2/jupiter ai2/ceres ai2/saturn \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --preemptible \
    --pure_docker_mode \
    --budget ai2/oe-adapt \
    --description "${RUN_NAME}" \
    --image "${BEAKER_IMAGE}" \
    --num_nodes 1 \
    --gpus $NUM_GPUS \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& \
    source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
    python open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --async_steps 2 \
    --beta 0.0 \
    --eval_pass_at_k 32 \
    --log_train_solve_rate_metrics \
    --load_ref_policy false \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list mnoukhov/manufactoria-qwen3-4b-instruct-pass128 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list mnoukhov/manufactoria-qwen3-4b-instruct-pass128 50 \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
    --apply_verifiable_reward true \
    --manufactoria_api_url \$MANUFACTORIA_API_URL/test_solution \
    --manufactoria_scoring_mode pass_rate \
    --temperature 1.0 \
    --total_episodes 768000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --clip_higher 0.28 \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 25 \
    --checkpoint_state_freq 25 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    "$@"
