#!/bin/bash
# VAPO (Value-model-based Augmented PPO) single GPU test script
# Scaled-down version of rlzero_seqlen.sh with VAPO value model
# Uses 1 GPU for quick local testing
# Reference: https://arxiv.org/abs/2504.05118

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

uv run python open_instruct/grpo_fast.py \
    --exp_name vapo_1gpu_test \
    --beta 0.0 \
    --async_steps 8 \
    --inflight_updates \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
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
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --use_value_model \
    --value_loss_coef 0.5 \
    --value_learning_rate 2e-6 \
    --vf_clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --value_warmup_steps 150 \
    --reset_optimizer_after_value_warmup \
    --decoupled_gae \
    --length_adaptive_gae \
    --length_adaptive_gae_alpha 0.05 \
    --loss_fn dapo \
    --clip_higher 0.28 \
    --push_to_hub false
