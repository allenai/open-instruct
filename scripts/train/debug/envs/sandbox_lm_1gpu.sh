#!/bin/bash
# Debug script for testing GRPO with SandboxLM environment (local 1 GPU)
#
# SandboxLM: Coding environment with execute_bash + str_replace_editor tools.
# System prompt and tools modified from https://github.com/llm-in-sandbox/llm-in-sandbox
#
# Requirements:
# - Docker running (uses python:3.12-slim image)
# - 1 GPU available
#
# Dataset: allenai/Dolci-RLZero-Math-7B — math tasks where the model can
# write and execute code in the sandbox to compute answers.

set -e

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting SandboxLM environment training (1 GPU)..."

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --total_episodes 80 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --apply_verifiable_reward true \
    --tools generic_sandbox \
    --pool_size 4 \
    --max_steps 30 \
    --tool_parser_type vllm_hermes \
    --system_prompt_override_file scripts/train/debug/envs/sandbox_lm_system_prompt.txt \
    --no_filter_zero_std_samples \
    --output_dir output/sandbox_lm_debug

echo "Training complete!"
