#!/bin/bash
# Debug script for testing GRPO with CounterEnv
#
# Simple counter environment: increment to reach target, then submit.
# No external dependencies required - uses built-in environment.

set -e

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting CounterEnv training (1 GPU, 80 episodes = 5 training steps)..."

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-counter-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 1536 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --total_episodes 80 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --max_steps 20 \
    --tool_parser_type vllm_hermes \
    --no_filter_zero_std_samples \
    --dataset_skip_cache \
    --output_dir output/counter_debug

echo "Training complete!"
