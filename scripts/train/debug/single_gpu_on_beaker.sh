#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --cluster ai2/ceres \
       --image "$BEAKER_IMAGE" \
       --description "Single GPU on Beaker test script." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --num_nodes 1 \
       --max_retries 0 \
       --timeout 15m \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --budget ai2/oe-adapt \
       --gpus 1 \
	   -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --inflight_updates True \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy true \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode
