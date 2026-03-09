#!/bin/bash
# Qwen3-4B-Base vanilla PPO on math data (100 steps value pretrain)
# 1 node (8 GPUs): 4 learner + 4 vLLM engines, 8k seqlen
# Requires hamish/ppo-value-model branch
DDMM=$(date +"%d%m")
exp_name=vip_ppo_${DDMM}_qwen3_4b_math_8k
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority high \
    --preemptible \
    --num_nodes 1 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env WANDB_RUN_ID=${exp_name}_$(date +%s) \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --beta 0.01 \
    --async_steps 8 \
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
    --kl_estimator 2 \
    --dataset_mixer_list allenai/Dolci-RLZero-Math-7B 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/Dolci-RLZero-Math-7B 16 \
    --dataset_mixer_eval_list_splits train train \
    --max_prompt_token_length 2048 \
    --response_length 6144 \
    --pack_length 8192 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name r1_simple_chat_postpend_think \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 4096000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --eval_on_step_0 True \
    --loss_fn dapo \
    --clip_higher 0.28 \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup \
    --whiten_advantages \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --vf_clip_range 0.2 \
    --decoupled_gae \
    --length_adaptive_gae
