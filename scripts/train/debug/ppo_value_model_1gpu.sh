#!/bin/bash
# Test script for PPO with value model feature in grpo_fast.py
# Uses 1 GPU for quick testing with Qwen3-0.6B

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --image "$BEAKER_IMAGE" \
    --description "PPO with value model test (1 GPU)" \
    --pure_docker_mode \
    --no-host-networking \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 15m \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --no_auto_dataset_cache \
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
    --model_name_or_path Qwen/Qwen3-0.6B \
    --chat_template_name tulu \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --learning_rate 1e-6 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --with_tracking \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --load_ref_policy false \
    --seed 1 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --push_to_hub false \
    --single_gpu_mode \
    --use_value_model \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --loss_fn dapo \
    --clip_higher 0.2
