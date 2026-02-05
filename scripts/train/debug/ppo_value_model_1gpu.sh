#!/bin/bash
# Test script for PPO with value model feature in grpo_fast.py
# Uses 1 GPU for quick local testing with Qwen3-0.5B
exp_name=ppo_value_model_1gpu_test
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --description "PPO with value model test (1 GPU)" \
    --timeout 1h \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 1 -- python open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --load_ref_policy false \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 2048 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --chat_template_name tulu \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 256 \
    --deepspeed_stage 3 \
    --num_learners_per_node 1 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 5 \
    --save_freq 50 \
    --gradient_checkpointing \
    --with_tracking \
    --use_value_model \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --loss_fn dapo \
    --clip_higher 0.2 \
    --push_to_hub False
