#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

# note: you might have to setup your own search api endpoint. I've been using massive-serve:
# https://github.com/RulinShao/massive-serve
# and then set the search_api_endpoint accordingly.
uv run python mason.py \
       --cluster ai2/jupiter-cirrascale-2 \
       --cluster ai2/augusta-google-1 \
       --cluster ai2/saturn-cirrascale \
       --image "$BEAKER_IMAGE" \
       --description "Single GPU on Beaker with tool use test script." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --gpus 1 \
	   -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k_with_tool_prompt 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k_with_tool_prompt 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --stop_strings "</answer>" \
    --apply_verifiable_reward true \
    --verbose True \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --exp_name 0605_general_tool_use_without_good_outputs \
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
    --output_dir /output \
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
    --tools code search \
    --search_api_endpoint "http://neptune-cs-aus-258.reviz.ai2.in:43189/search" \
    --code_tool_api_endpoint https://open-instruct-tool-server-10554368204.us-central1.run.app/execute
