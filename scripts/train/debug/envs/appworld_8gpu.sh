#!/bin/bash
# Beaker experiment for AppWorldEnv — 8 GPUs (4 learners, 4 inference engines)
#
# AppWorld environment with stateful Python execution through appworld_execute.
# Uses Qwen3-4B-Instruct and train/eval datasets built by:
#   scripts/data/create_appworld_env_datasets.py
#
# Requirements:
# - Network access in the container to fetch AppWorld artifacts on first run
# - Optional: set APPWORLD_ROOT to override the in-container install path
#
# Usage: bash scripts/train/build_image_and_launch.sh scripts/train/debug/envs/appworld_8gpu.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

APPWORLD_ROOT="${APPWORLD_ROOT:-/root/.appworld}"
APPWORLD_TRAIN_DATASET="${APPWORLD_TRAIN_DATASET:-hamishivi/rlenv-appworld-train}"
APPWORLD_EVAL_DATASET="${APPWORLD_EVAL_DATASET:-hamishivi/rlenv-appworld-eval}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "AppWorldEnv 8-GPU (4L/4E) Qwen3-4B-Instruct" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --preemptible \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --env APPWORLD_ROOT="$APPWORLD_ROOT" \
       --budget ai2/oe-adapt \
       --gpus 8 \
       --no_auto_dataset_cache \
       -- bash configs/beaker_configs/appworld_setup.sh \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list "${APPWORLD_TRAIN_DATASET}" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "${APPWORLD_EVAL_DATASET}" 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 3072 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path "${MODEL_NAME}" \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --lr_scheduler_type constant \
    --total_episodes 20480 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --inflight_updates True \
    --async_steps 1 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub false \
    --save_traces \
    --tools appworld \
    --max_steps 40 \
    --per_turn_max_tokens 1024 \
    --tool_parser_type vllm_hermes \
    --reward_aggregator last \
    --no_filter_zero_std_samples \
    --local_eval_every 10 \
    --eval_on_step_0 true \
    --dataset_skip_cache \
    --vllm_enable_prefix_caching \
    --output_dir output/appworld_8gpu
