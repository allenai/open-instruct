#!/bin/bash

set -euo pipefail

# Start a local single-process Code API server and set CODE_API_URL
PORT=${PORT:-8070}
LOG_FILE=${LOG_FILE:-/tmp/code_api.log}

echo "Starting local Code API server on 127.0.0.1:${PORT}..."
nohup uvicorn open_instruct.code_utils.api:app --host 127.0.0.1 --port ${PORT} > "${LOG_FILE}" 2>&1 &
API_PID=$!
echo "Code API PID: ${API_PID} (logs: ${LOG_FILE})"

# Wait briefly for the server to become ready
for i in {1..20}; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null; then
        break
    fi
    sleep 0.3
done

export CODE_API_URL="http://127.0.0.1:${PORT}"
echo "CODE_API_URL=${CODE_API_URL}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Ensure a local writable output directory
mkdir -p output

python open_instruct/grpo_fast.py \
    --dataset_mixer_list saurabh5/rlvr-code-view-tool-new-first-turn-only-user 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list saurabh5/rlvr-code-view-tool-new-first-turn-only-user 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --exp_name debug_code_view_sft_1gpu \
    --learning_rate 5e-7 \
    --total_episodes 3_200 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 1 \
    --local_eval_every 10 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --output_dir ./output \
    --kl_estimator kl3 \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --update_progress_every 1 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --vllm_num_engines 1 \
    --max_tool_calls 5 \
    --vllm_enable_prefix_caching \
    --tools code_agent \
    --code_agent_api_endpoint \$CODE_API_URL/view_file \
    --allow_world_padding True
