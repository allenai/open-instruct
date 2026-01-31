#!/bin/bash
# Single GPU debug: Wordle env via OpenEnv/TextArena + python tool
#
# Dataset: hamishivi/wordle_env_train
#   - Uses dataset: "env" for EnvVerifier (reward from env.step())
#
# This script starts a local TextArena server and connects to it.
#
# Model: Qwen3-0.6B with /no_think mode (optimal for base model training)
#   - Fixed tool schema (includes "word" parameter)
#   - No-think mode prevents over-verbosity

set -e

# Configuration
TEXTARENA_PORT=${TEXTARENA_PORT:-8765}
TEXTARENA_URL="http://localhost:${TEXTARENA_PORT}"

# Start TextArena server in background
echo "Starting TextArena Wordle server on port ${TEXTARENA_PORT}..."
TEXTARENA_ENV_ID=Wordle-v0 uv run --extra openenv \
    python -m uvicorn textarena_env.server.app:app \
    --host 0.0.0.0 --port ${TEXTARENA_PORT} &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s "${TEXTARENA_URL}/health" > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Cleanup function
cleanup() {
    echo "Stopping TextArena server..."
    kill $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Run training
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run --extra openenv open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/wordle_expert_train 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/wordle_expert_train 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 1536 \
    --pack_length 3072 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --system_prompt_override_file no_think_system_prompt.txt \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --learning_rate 3e-7 \
    --lr_scheduler_type constant \
    --total_episodes 320 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --single_gpu_mode \
    --vllm_gpu_memory_utilization 0.55 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools wordle python \
    --tool_call_names wordle code \
    --tool_configs "{\"base_url\": \"${TEXTARENA_URL}\", \"pool_size\": 1}" '{"api_endpoint": "https://open-instruct-tool-server.run.app/execute", "timeout": 3}' \
    --tool_parser_type vllm_hermes \
    --max_tool_calls 10 \
    --filter_zero_std_samples false \
    --push_to_hub false
