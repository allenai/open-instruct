#!/bin/bash
# Debug script for testing GRPO with SWERL Sandbox environment (local 1 GPU)
#
# SWERL Sandbox: Per-sample Docker tasks with submit-based evaluation.
# Extends GenericSandboxEnv with per-task instruction, seeds, and test scripts.
#
# Requirements:
# - Docker running (uses python:3.12-slim image by default)
# - 1 GPU available
# - Test data at data/swerl_sandbox_test/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting SWERL Sandbox environment training (1 GPU)..."

cd "$REPO_ROOT"
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/agent-task-combined 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 1.0 \
    --learning_rate 3e-7 \
    --total_episodes 16 \
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
    --tools swerl_sandbox \
    --tool_configs '{"task_data_dir": "'"$REPO_ROOT"'/data/swerl_sandbox_test", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 4 \
    --max_steps 20 \
    --tool_parser_type vllm_hermes \
    --no_filter_zero_std_samples \
    --dataset_skip_cache \
    --output_dir output/swerl_sandbox_debug

echo "Training complete!"
