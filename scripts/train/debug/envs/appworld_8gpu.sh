#!/bin/bash
# Debug script for testing GRPO with AppWorld environment
#
# This script runs an 8-GPU training session with the AppWorld environment.
# No Docker, E2B, or external server required - uses AppWorld library directly.
#
# Requirements:
# - 8 GPUs available
# - AppWorld installed: pip install appworld && appworld install && appworld download data
#
# See: https://github.com/stonybrooknlp/appworld

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

# Check AppWorld is installed
if ! python -c "import appworld" 2>/dev/null; then
    echo "ERROR: AppWorld not installed."
    echo "Install with:"
    echo "  pip install appworld"
    echo "  appworld install"
    echo "  appworld download data"
    exit 1
fi
echo "AppWorld found!"

echo "Starting AppWorld training (8 GPU, 5 episodes)..."

cd "$REPO_ROOT"
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$REPO_ROOT/data/envs/appworld_train.jsonl" 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 2048 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --temperature 0.7 \
    --learning_rate 1e-6 \
    --total_episodes 5 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_tensor_parallel_size 8 \
    --beta 0.01 \
    --seed 42 \
    --vllm_sync_backend nccl \
    --vllm_gpu_memory_utilization 0.8 \
    --gradient_checkpointing \
    --push_to_hub false \
    --save_traces \
    --env_name appworld \
    --env_pool_size 32 \
    --env_max_steps 50 \
    --env_timeout 300 \
    --tool_parser_type vllm_hermes \
    --output_dir output/appworld_debug

echo "Training complete!"
