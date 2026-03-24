#!/bin/bash
# Qwen2.5-3B-Instruct PPO with lambda=1.0
# 1 node (8 GPUs): 4 learner + 4 vLLM engines
DDMM=$(date +"%d%m")
exp_name=vip_ppo_instruct_lam1_${DDMM}_qwen25_3b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
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
    --no_resampling_pass_rate 0.875 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
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
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
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
    --value_learning_rate 1e-5 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --value_loss_coef 0.5 \
    --gamma 1.0 \
    --gae_lambda 1.0 \
    --vf_clip_range 0.2
