#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

# Source the virtual environment first
source .venv/bin/activate

python local_grpo_test.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k_with_tool_prompt 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k_with_tool_prompt 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --async_steps 0 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --exp_name 0605_general_tool_use_without_good_outputs \
    --learning_rate 5e-7 \
    --total_episodes 500000 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --output_dir /output \
    --kl_estimator kl3 \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --vllm_num_engines 1 \
    --max_tool_calls 5 \
    --vllm_enable_prefix_caching \
    --tools code search \
    --search_api_endpoint "http://saturn-cs-aus-232.reviz.ai2.in:44177/search" \
    --code_tool_api_endpoint https://open-instruct-tool-server-10554368204.us-central1.run.app/execute
