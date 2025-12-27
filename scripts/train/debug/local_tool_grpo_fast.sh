#!/bin/bash
# Local tool use training script with code execution and search
#
# Note: replace with your own key if outside of ai2.
export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/olmo-instruct)

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
    --max_prompt_token_length 512 \
    --response_length 512 \
    --active_sampling \
    --async_steps 8 \
    --pack_length 1024 \
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
    --tools serper_search python \
    --tool_tag_names search code \
    --code_api_endpoint "$CODE_SERVER_ENDPOINT" \
    --push_to_hub false
    # To swap search backend while keeping same tags, change to:
    # --tools s2_search code \
    # --tool_tag_names search code \
