#!/bin/bash

# Single-GPU local smoke test for SubTB+GM with a separate flow/value model.
# Uses a small Qwen3 base checkpoint to validate trainer wiring cheaply.

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

uv run python open_instruct/grpo_fast.py \
    --exp_name qwen3_0p6b_subtb_1gpu_test \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 1024 \
    --pack_length 1536 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --chat_template qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 0.6666667 \
    --total_episodes 256 \
    --deepspeed_stage 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 5 \
    --gradient_checkpointing \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --single_gpu_mode \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --load_ref_policy True \
    --push_to_hub false \
    --use_value_model \
    --value_loss_type subtb_gm \
    --value_learning_rate 2e-6 \
    --value_warmup_steps 50 \
    --reset_optimizer_after_value_warmup \
    --value_loss_coef 0.5 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --decoupled_gae \
    --length_adaptive_gae \
    --length_adaptive_gae_alpha 0.05 \
    --subtb_q 0.5 \
    --subtb_alpha 1.0 \
    --subtb_omega 1.0 \
    --subtb_reward_scale 15.0 \
    --subtb_num_windows 8 \
    --subtb_min_window_size 16 \
    --subtb_max_window_size 256 \
    --subtb_lambda 0.9 "$@"
