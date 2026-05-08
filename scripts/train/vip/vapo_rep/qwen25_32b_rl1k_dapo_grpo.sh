#!/bin/bash
# RL: vanilla GRPO baseline matched to the verl DAPO Qwen2.5-32B reproduction
# (https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/runs/0qjd0wap),
# with no value model, no SAE, and no answer-prefix conditioning.
# Settings:
#   * 512 prompts/step, 16 samples/prompt, mini-batch 32 prompts (=> 16 mb)
#   * clip_lower=0.20, clip_higher=0.28 (DAPO clip-higher), token-mean DAPO loss
#   * lr=1e-6, weight_decay=0.1, grad_clip=1.0, entropy 0
#   * dynamic sampling (active_sampling), max_prompt 2048, response 20480
#   * temperature 1.0, top_p 1.0, gen TP=4
# DAPO math dataset (hamishivi/DAPO-Math-17k-Processed_filtered).
#
# GPU layout: 4 nodes x 8 GPUs = 16 learner GPUs (2 nodes) + 16 vLLM GPUs (2
# nodes). vLLM uses TP=2 (8 engines) and the trainer uses SP=4 to fit
# Qwen2.5-32B with the long generation budget.
#
# Step budget:
#   total_episodes = 1000 RL steps * 512 prompts * 16 samples
#                  = 8192000
DDMM=$(date +"%d%m")
exp_name=vip_rl1k_dapo_grpo_${DDMM}_qwen25_32b_math
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
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
    --total_episodes 8192000 \
    --deepspeed_stage 3 \
    --deepspeed_offload_param \
    --deepspeed_offload_optimizer \
    --gather_whole_model False \
    --num_learners_per_node 8 8 \
    --sequence_parallel_size 4 \
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
    --push_to_hub False
