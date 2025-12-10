#!/bin/bash
# Debug script for on-policy distillation training
#
# This script runs a minimal distillation training job for testing.
# Before running, ensure you have a teacher model server running, e.g.:
#   vllm serve Qwen/Qwen3-1.7B --port 8000
#
# Usage:
#   ./scripts/train/debug/distillation.sh

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

uv run python open_instruct/on_policy_distillation.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 1 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --stop_strings "</answer>" \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 100 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 1.0 \
    --seed 42 \
    --local_eval_every 10 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --teacher_api_base http://localhost:8000/v1 \
    --teacher_api_key EMPTY \
    --teacher_max_concurrent_requests 32 \
    --teacher_timeout 120.0 \
    --teacher_temperature 1.0
