#!/bin/bash
# Debug script for testing GRPO with AppWorld environment (local 1 GPU)
#
# AppWorld: Interactive environment with 9 apps (Spotify, Amazon, Venmo, etc.)
# and 457 APIs. The model executes Python code to complete tasks.
#
# Requirements:
# - appworld package installed (included in project dependencies)
# - AppWorld data already in repo at data/ (base_dbs, tasks, api_docs)
# - 1 GPU available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export APPWORLD_ROOT="$REPO_ROOT"

echo "Starting AppWorld environment training (1 GPU)..."
echo "Note: AppWorld has 733 tasks across 9 apps (Spotify, Amazon, Venmo, etc.)"

cd "$REPO_ROOT"
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-appworld-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 80 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --backend_timeout 300 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --env_pool_size 4 \
    --env_max_steps 20 \
    --tool_parser_type vllm_hermes \
    --no_filter_zero_std_samples \
    --output_dir output/appworld_debug

echo "Training complete!"
