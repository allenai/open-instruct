#!/bin/bash
# 1-GPU test script for the generic_mcp tool using a weather MCP server
#
# This script tests the generic_mcp tool integration by:
# 1. Starting a local weather MCP server
# 2. Running GRPO training with the generic_mcp tool pointing to the server
#
# Usage:
#   ./scripts/train/debug/generic_mcp_weather_test.sh
#
# The weather server provides mock weather data for testing the MCP tool
# without requiring external API keys.

set -e  # Exit on error

# Configuration
WEATHER_SERVER_PORT=${WEATHER_SERVER_PORT:-8765}
WEATHER_SERVER_HOST="0.0.0.0"

# Start the weather MCP server in the background
echo "Starting weather MCP server on port $WEATHER_SERVER_PORT..."
cd open_instruct/tools/weather_mcp_server
uv run python server.py $WEATHER_SERVER_PORT &
WEATHER_SERVER_PID=$!
cd - > /dev/null

# Wait for server to start
echo "Waiting for weather server to start..."
sleep 3

# Verify server is running
if ! kill -0 $WEATHER_SERVER_PID 2>/dev/null; then
    echo "ERROR: Weather server failed to start"
    exit 1
fi

echo "Weather server started (PID: $WEATHER_SERVER_PID)"

# Cleanup function to kill server on exit
cleanup() {
    echo "Stopping weather MCP server (PID: $WEATHER_SERVER_PID)..."
    kill $WEATHER_SERVER_PID 2>/dev/null || true
    wait $WEATHER_SERVER_PID 2>/dev/null || true
    echo "Weather server stopped."
}
trap cleanup EXIT

# The MCP endpoint URL
MCP_ENDPOINT="http://localhost:$WEATHER_SERVER_PORT/mcp"
echo "MCP endpoint: $MCP_ENDPOINT"

# Run training with the generic_mcp tool
echo "Starting GRPO training with generic_mcp tool..."
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run --extra mcp python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/wots_the_weather 5000 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/wots_the_weather 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --active_sampling \
    --async_steps 8 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools generic_mcp \
    --tool_configs '{"server_url": "'"$MCP_ENDPOINT"'", "timeout": 30}' \
    --tool_parser vllm_hermes \
    --push_to_hub false

echo "Training completed successfully!"
