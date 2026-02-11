#!/bin/bash
# VAPO (Value-model-based Augmented PPO) single GPU test script
# Scaled-down version of rlzero_seqlen.sh with VAPO value model
# Uses 1 GPU for quick testing
# Reference: https://arxiv.org/abs/2504.05118

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --cluster ai2/ceres \
    --image "$BEAKER_IMAGE" \
    --description "VAPO test (1 GPU)" \
    --pure_docker_mode \
    --no-host-networking \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout 30m \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --budget ai2/oe-adapt \
    --gpus 1 \
    --no_auto_dataset_cache \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name vapo_1gpu_test \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/Dolci-RLZero-Math-7B 16 \
    --dataset_mixer_eval_list_splits train train \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 2048 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --chat_template_name tulu \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 256 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 5 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --use_value_model \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --value_warmup_steps 50 \
    --decoupled_gae \
    --length_adaptive_gae \
    --length_adaptive_gae_alpha 0.05 \
    --loss_fn dapo \
    --clip_higher 0.28 \
    --push_to_hub false
