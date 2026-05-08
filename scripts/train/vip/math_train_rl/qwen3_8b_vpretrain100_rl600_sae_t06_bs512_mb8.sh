#!/bin/bash
# Value warmup + RL: SAE-only (no GT conditioning) with scaled vLLM logprobs
# on Qwen3-8B-Base.
# 100 value-only warmup steps, then 600 regular policy/value RL steps.
#
# GPU layout: 2 nodes x 8 GPUs = 8 learner GPUs + 8 vLLM inference GPUs.
#
# Differences from the canonical vpretrain100_rl1k SAE scripts:
#   - 8 samples / 64 prompts => 512 samples per batch
#   - 8 minibatches per step
#   - sampling temperature 0.6 (vLLM logprobs are scaled by 1/temperature in
#     the data loader so that they line up with the trainer's
#     temperature-scaled log_softmax)
#   - 600 RL steps after the 100-step value warmup
#
# Step budget:
#   total_episodes = (100 value warmup + 600 RL) * 64 prompts * 8 samples
#                  = 358400
DDMM=$(date +"%d%m")
exp_name=vip_vpretrain100_rl600_sae_t06_bs512_mb8_${DDMM}_qwen3_8b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 2 \
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
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 64 \
    --num_mini_batches 8 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 0.6 \
    --total_episodes 358400 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 8 \
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
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --gae_lambda 0.95 \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --use_sae \
    --sae_threshold 0.2 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup
