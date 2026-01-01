#!/bin/bash
# 8-gpu training script for code and search tools with OLMo 3.

# Note: replace with your own key if outside of ai2.
export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/dr-tulu-ablations)

# Check if we're using local code server or remote
USE_LOCAL_CODE_SERVER=${USE_LOCAL_CODE_SERVER:-false}
CODE_SERVER_ENDPOINT=${CODE_SERVER_ENDPOINT:-https://open-instruct-tool-server-10554368204.us-central1.run.app/execute}

if [ "$USE_LOCAL_CODE_SERVER" = true ]; then
    # Start the code server in the background
    echo "Starting code execution server on port 1212..."
    cd open_instruct/tools/code_server
    uv run uvicorn server:app --host 0.0.0.0 --port 1212 &
    CODE_SERVER_PID=$!
    cd - > /dev/null

    # Wait for server to start
    sleep 3
    CODE_SERVER_ENDPOINT="http://0.0.0.0:1212/execute"

    # Cleanup function to kill server on exit
    cleanup() {
        echo "Stopping code server (PID: $CODE_SERVER_PID)..."
        kill $CODE_SERVER_PID 2>/dev/null
    }
    trap cleanup EXIT
fi

# Run training
VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --active_sampling \
    --inflight_updates \
    --async_steps 8 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --learning_rate 3e-7 \
    --total_episodes 2000 \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 6 \
    --vllm_tensor_parallel_size 2 \
    --vllm_num_engines 1 \
    --beta 0.00 \
    --seed 3 \
    --local_eval_every 1 \
    --eval_on_step_0 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --tools serper_search python \
    --tool_tag_names search code \
    --tool_parser vllm_olmo3 \
    --code_api_endpoint "$CODE_SERVER_ENDPOINT" \
    --push_to_hub false

