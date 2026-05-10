#!/bin/bash
# Value warmup + RL: scalar PPO critic with decoupled, length-adaptive GAE,
# DAPO clip-higher and DAPO loss, for 100 value-only steps then 1000 regular
# policy/value RL steps on Qwen2.5-32B-Base.
# Hyperparameters mirror the verl DAPO Qwen2.5-32B reproduction
# (https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/runs/0qjd0wap):
#   * 512 prompts/step, 16 samples/prompt, mini-batch 32 prompts (=> 16 mb)
#   * clip_lower=0.20, clip_higher=0.28 (DAPO clip-higher), token-mean DAPO loss
#   * lr=1e-6, weight_decay=0.1, grad_clip=1.0, entropy 0
#   * dynamic sampling (active_sampling), max_prompt 2048, response 20480
#   * temperature 1.0, top_p 1.0, gen TP=4
# DAPO math dataset (hamishivi/DAPO-Math-17k-Processed_filtered).
# No SAE / no answer-prefix conditioning -- this is the plain PPO + decoupled
# length-adaptive GAE variant.
#
# GPU layout: 8 nodes x 8 GPUs = 48 learner GPUs (6 nodes) + 16 vLLM GPUs (2
# nodes). vLLM uses TP=2 (8 engines) and the trainer uses SP=2 (DP=24) to fit
# Qwen2.5-32B with the long generation budget on-GPU (no offload).
#
# Step budget:
#   total_episodes = (100 value warmup + 1000 RL) * 512 prompts * 16 samples
#                  = 9011200
DDMM=$(date +"%d%m")
exp_name=vip_vpretrain100_rl1k_ppo_decoupled_lagae_${DDMM}_qwen25_32b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 8 \
    --gpus 8 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_NCCL_ENABLE_MONITORING=0 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
    --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
    --env WANDB_RUN_ID=${exp_name}_$(date +%s) \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& uv run hf download Qwen/Qwen2.5-32B --include "*.json" "merges.txt" --max-workers 1 \&\& python open_instruct/grpo_fast.py \
    --exp_name ${exp_name} \
    --beta 0.0 \
    --backend_timeout 240 \
    --async_steps 8 \
    --inflight_updates \
    --no_resampling_pass_rate 0.875 \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --active_sampling \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 512 \
    --num_mini_batches 16 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --clip_lower 0.2 \
    --clip_higher 0.28 \
    --loss_fn dapo \
    --loss_denominator token \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 20480 \
    --pack_length 22528 \
    --model_name_or_path Qwen/Qwen2.5-32B \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --total_episodes 9011200 \
    --deepspeed_stage 3 \
    --deepspeed_zpg 1 \
    --gather_whole_model False \
    --num_learners_per_node 8 8 8 8 8 8 \
    --sequence_parallel_size 2 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 20 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub False \
    --use_value_model \
    --value_learning_rate 2e-6 \
    --gae_lambda 0.95 \
    --decoupled_gae \
    --length_adaptive_gae \
    --gamma 1.0 \
    --value_loss_coef 0.5 \
    --vf_clip_range 0.2 \
    --value_warmup_steps 100 \
    --reset_optimizer_after_value_warmup
