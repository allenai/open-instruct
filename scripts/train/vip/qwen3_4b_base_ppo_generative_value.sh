#!/bin/bash
# Qwen3-4B-Base generative value model pretraining (1000 steps)
# Trains only the generative value model via REINFORCE (reward = 1 - |pred - outcome|)
# Policy LR=0 so the policy is frozen — only the generative value model learns
# 1 node (8 GPUs): 4 learner + 4 vLLM engines, 32 prompts x 16 rollouts
DDMM=$(date +"%d%m")
exp_name=vip_genvalue_pretrain_${DDMM}_qwen3_4b_math
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
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 0.0 \
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
    --total_episodes 512000 \
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
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    --use_generative_value_model \
    --value_learning_rate 5e-6 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --generative_value_chunk_size 512 \
    --generative_value_max_think_tokens 256 \
    --generative_value_reinforce_coef 0.1
