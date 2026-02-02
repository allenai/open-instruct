#!/bin/bash
# Debug script for testing GRPO with GuessNumberEnv
#
# Number guessing game: guess a secret number between 1 and 100.
# No external dependencies required - uses built-in environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

echo "Starting GuessNumberEnv training (1 GPU, 5 episodes)..."

cd "$REPO_ROOT"
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list "$REPO_ROOT/data/envs/guess_number_train.jsonl" 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 256 \
    --pack_length 768 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 5 \
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
    --env_name guess_number \
    --env_pool_size 4 \
    --env_max_steps 10 \
    --tool_parser_type vllm_hermes \
    --output_dir output/guess_number_debug

echo "Training complete!"
