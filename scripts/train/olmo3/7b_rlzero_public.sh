#!/bin/bash

MODEL_NAME_OR_PATH="allenai/Olmo-3-1025-7B"
DATASETS="allenai/Dolci-RLZero-Math-7B 1.0"

# AIME 2024, 2025 local single-sample evals
# Full, bootstrapped pass@32 evals must be run separately
LOCAL_EVALS="allenai/aime2024-25-rlvr 1.0 allenai/aime2024-25-rlvr 1.0"
LOCAL_EVAL_SPLITS="test_2024 test_2024 test_2025 test_2025"

EXPERIMENT_NAME="olmo3-7b_rlzero"

python open_instruct/grpo_fast.py \
    --exp_name ${EXPERIMENT_NAME} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_rlzero \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --no_resampling_pass_rate 0.875 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --active_sampling \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512256 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 32 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --eval_on_step_0 True
