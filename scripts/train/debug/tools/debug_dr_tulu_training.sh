#!/bin/bash
# Debug script for DR Tulu style tool use training (1-GPU configuration)
#
# This script demonstrates training with DR Tulu MCP tools for research assistance.
# It uses the dr_agent package which provides access to various search tools.
#
# Prerequisites:
#   1. Install dr_agent: uv sync --extra dr-tulu
#   2. Set required API keys (see below)
#   3. Start MCP server (done automatically by this script)
#
# API Keys needed (set these environment variables or use beaker secrets):
#   - SERPER_API_KEY: For Google search via Serper API
#   - S2_API_KEY: For Semantic Scholar snippet search
#   - CRAWL4AI_API_KEY: For webpage browsing (optional)

set -e

# =============================================================================
# Configuration
# =============================================================================

# MCP server configuration
MCP_PORT=${MCP_PORT:-8000}
MCP_HOST=${MCP_HOST:-0.0.0.0}

# API Keys - uncomment and set your own, or use beaker secrets
# export SERPER_API_KEY="your-serper-api-key"
# export S2_API_KEY="your-semantic-scholar-api-key"

# Optional: Beaker secret loading (AI2 internal)
# export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/dr-tulu-ablations)
# export S2_API_KEY=$(beaker secret read hamishivi_S2_API_KEY --workspace ai2/dr-tulu-ablations)

# =============================================================================
# Setup
# =============================================================================

echo "Installing dr-tulu dependencies..."
uv sync --extra dr-tulu

echo "Starting MCP server on ${MCP_HOST}:${MCP_PORT}..."
MCP_CACHE_DIR=".cache-$(hostname)" uv run --extra dr-tulu python -m dr_agent.mcp_backend.main \
    --port "$MCP_PORT" \
    --host "$MCP_HOST" \
    --path /mcp &
MCP_PID=$!

# Wait for server to start
sleep 5

# Cleanup function
cleanup() {
    echo "Stopping MCP server..."
    kill $MCP_PID 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# Training
# =============================================================================

echo "Starting DR Tulu training..."
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run --extra dr-tulu open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think_tool_vllm \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --system_prompt_override_file scripts/train/debug/tools/dr_tulu_system_prompt.txt \
    --tools mcp \
    --tool_parser dr_tulu \
    --tool_configs '{"tool_names": "snippet_search,google_search", "parser_name": "unified", "host": "'"$MCP_HOST"'", "port": '"$MCP_PORT"'}' \
    --max_tool_calls 5 \
    --push_to_hub false

echo "Training complete!"
