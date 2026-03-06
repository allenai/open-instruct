#!/bin/bash
# Debug script for testing GRPO with SWERL Sandbox environment (local 1 GPU)
# Uses Qwen3.5-0.8B (requires vllm nightly >=0.17.0rc1)
#
# Requirements:
# - Docker running (uses python:3.12-slim image by default)
# - 1 GPU available

set -e

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting SWERL Sandbox environment training (1 GPU, Qwen3.5-0.8B)..."

# Ensure venv is synced first, then force-upgrade transformers (vLLM caps <5 but Qwen3.5 needs >=5.3.0)
uv sync
uv pip install --upgrade "huggingface_hub>=0.33.0" "tokenizers>=0.21"
uv pip install --no-deps "transformers>=5.3.0"

uv run --no-sync python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/agent-task-combined 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
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
    --tool_configs '{"task_data_hf_repo": "hamishivi/agent-task-combined", "test_timeout": 120, "image": "python:3.12-slim"}' \
    --pool_size 4 \
    --max_steps 10 \
    --tool_parser_type vllm_qwen3_xml \
    --no_filter_zero_std_samples \
    --dataset_skip_cache \
    --output_dir output/swerl_sandbox_qwen35_debug

echo "Training complete!"
