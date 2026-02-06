#!/bin/bash
# Debug script for testing GRPO with Wordle environment via OpenEnv
#
# This script runs a 1-GPU training session with the Wordle environment.
# The Wordle server is started automatically via OpenEnv.
#
# Requirements:
# - openenv package installed (pip install openenv)
# - 1 GPU available

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORDLE_PORT=8765
WORDLE_URL="http://localhost:$WORDLE_PORT"
WORDLE_PID=""

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

# Cleanup function to stop the server
cleanup() {
    if [ -n "$WORDLE_PID" ]; then
        echo "Stopping Wordle OpenEnv server (PID: $WORDLE_PID)..."
        kill "$WORDLE_PID" 2>/dev/null || true
        wait "$WORDLE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Check if Wordle server is running, start it if not
if curl -s "$WORDLE_URL/reset" -X POST -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; then
    echo "Wordle OpenEnv server is already running at $WORDLE_URL"
else
    echo "Starting Wordle OpenEnv server (textarena_env)..."
    uv run python -c "from textarena_env.server.app import main; main(port=$WORDLE_PORT)" > /dev/null 2>&1 &
    WORDLE_PID=$!

    # Wait for server to be ready (max 30 seconds)
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s "$WORDLE_URL/reset" -X POST -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; then
            echo "Wordle OpenEnv server started successfully (PID: $WORDLE_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "ERROR: Wordle server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
fi

echo "Starting Wordle environment training (1 GPU, 24 episodes = 3 training steps)..."

cd "$REPO_ROOT"
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/rlenv-wordle-nothink 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --total_episodes 48 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --vllm_sync_backend gloo \
    --backend_timeout 180 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --save_traces \
    --env_name openenv_text \
    --env_base_url "$WORDLE_URL" \
    --env_pool_size 4 \
    --env_max_steps 6 \
    --tool_parser_type vllm_hermes \
    --no_filter_zero_std_samples \
    --dataset_skip_cache \
    --output_dir output/wordle_debug

echo "Training complete!"
