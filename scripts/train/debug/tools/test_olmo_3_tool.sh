#!/bin/bash
# Multinode test script for GRPO with tool use using OLMo-3-7B-Instruct-SFT
# 1 training node (8 GPUs) + 1 inference node (8 GPUs)
# Note: Currently only 'code' tool is supported with new tools system

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "OLMo-3 multinode tool use test" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --timeout 1h \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --secret SERPER_API_KEY=hamishivi_SERPER_API_KEY \
       --gpus 8 \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/tulu_3_rewritten_100k 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --inflight_updates True \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --exp_name olmo3_7b_tool_use_test \
    --learning_rate 5e-7 \
    --total_episodes $((5 * 32 * 4)) \
    --deepspeed_stage 3 \
    --sequence_parallel_size 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 1 \
    --local_eval_every 10 \
    --gradient_checkpointing \
    --push_to_hub false \
    --output_dir /output \
    --kl_estimator 2 \
    --non_stop_penalty False \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --try_launch_beaker_eval_jobs_on_weka False \
    --max_tool_calls 5 \
    --vllm_enable_prefix_caching \
    --tools python serper_search \
    --tool_call_names code search \
    --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 3}' '{}' \
    --tool_parser_type vllm_olmo3
