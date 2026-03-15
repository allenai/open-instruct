#!/bin/bash
# Experiment (2): Warmup 100 steps, train policy 100 steps, then re-warmup 1000 steps
# Tests whether the value model learns better after the policy has improved.
# Steps 1-100: value warmup (policy frozen, value trains)
# Steps 101-200: normal PPO (both train)
# Steps 201-1200: value re-warmup (policy frozen again, value trains on improved policy)
DDMM=$(date +"%d%m")
exp_name=vip_ppo_rewarmup_${DDMM}_qwen3_4b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_RUN_ID=${exp_name}_$(date +%s) \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --async_steps 8 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --no_resampling_pass_rate 0.875 \
    --active_sampling \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 153600 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_top_p 1.0 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --eval_on_step_0 True \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --oe_eval_max_length 10240 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks aime:zs_cot_r1::pass_at_32_2024_rlzero,aime:zs_cot_r1::pass_at_32_2025_rlzero \
    --oe_eval_gpu_multiplier 4 \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --value_warmup_steps 100 \
    --value_rewarmup_start 201 \
    --value_rewarmup_steps 1000 \
    --reset_optimizer_after_value_warmup \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --decoupled_gae \
    --length_adaptive_gae
