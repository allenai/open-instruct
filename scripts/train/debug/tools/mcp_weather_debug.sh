#!/bin/bash
# Debug script for testing GRPO with generic MCP tool using the weather MCP server
#
# Dataset: https://huggingface.co/datasets/hamishivi/wots_the_weather
# This dataset contains weather-related questions that the model can answer
# using the get_current_weather, get_weather_forecast, and compare_weather tools.
#
# The weather MCP server provides 3 tools:
# - get_current_weather(city): Get current weather for a city
# - get_weather_forecast(city, days): Get a multi-day weather forecast
# - compare_weather(city1, city2): Compare weather between two cities
#
# The script will automatically start the weather server if it's not running.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WEATHER_SERVER_DIR="$REPO_ROOT/open_instruct/environments/tools/servers/weather_mcp_server"
WEATHER_SERVER_PORT=8765
WEATHER_SERVER_URL="http://localhost:$WEATHER_SERVER_PORT/mcp"
WEATHER_SERVER_PID=""

# Cleanup function to stop the server if we started it
cleanup() {
    if [ -n "$WEATHER_SERVER_PID" ]; then
        echo "Stopping weather MCP server (PID: $WEATHER_SERVER_PID)..."
        kill "$WEATHER_SERVER_PID" 2>/dev/null || true
        wait "$WEATHER_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Check if weather server is running, start it if not
if curl -s "$WEATHER_SERVER_URL" > /dev/null 2>&1; then
    echo "Weather MCP server is already running at $WEATHER_SERVER_URL"
else
    echo "Weather MCP server not running. Starting it..."
    cd "$WEATHER_SERVER_DIR"
    uv run python server.py "$WEATHER_SERVER_PORT" > /dev/null 2>&1 &
    WEATHER_SERVER_PID=$!
    cd "$REPO_ROOT"

    # Wait for server to be ready (max 30 seconds)
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s "$WEATHER_SERVER_URL" > /dev/null 2>&1; then
            echo "Weather MCP server started successfully (PID: $WEATHER_SERVER_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "ERROR: Weather MCP server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
fi

echo "Starting training..."

VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/wots_the_weather 32 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/wots_the_weather 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 1536 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --learning_rate 3e-7 \
    --total_episodes 100 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 42 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools generic_mcp \
    --tool_configs '{"server_url": "http://localhost:8765/mcp", "transport": "http", "timeout": 30}' \
    --tool_parser_type vllm_hermes \
    --max_tool_calls 3 \
    --verbose true \
    --push_to_hub false \
    --output_dir output/weather_mcp_debug \
    --dataset_skip_cache
