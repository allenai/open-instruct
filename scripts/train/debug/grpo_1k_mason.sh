#!/bin/bash
# Matched 1000-step GRPO baseline for scripts/train/debug/genac_1k_mason.sh.
#
# Same policy model, dataset, batch size, reward scale (verification_reward=1),
# response length, learning rate, KL coefficient, and evaluation/checkpoint cadence
# as the GenAC 1k script, but with the value-model / generative-critic path removed.
# This isolates whether GenAC's learned value signal improves over the usual
# group-relative advantages.
#
# GPU layout (6 GPUs on 1 node):
#   - 4x policy learners (DeepSpeed stage 3)
#   - 2x policy vLLM engines (TP=1)
#
# Step budget:
#   total_episodes = 1000 * num_unique_prompts_rollout * num_samples_per_prompt_rollout
#                  = 1000 * 8 * 16 = 128000
DDMM=$(date +"%d%m")
exp_name=grpo_1k_${DDMM}_qwen3_4b_math
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
    --gpus 6 \
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
    --active_sampling \
    --no_resampling_pass_rate 0.875 \
    --loss_fn dapo \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8096 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --chat_template_name qwen_instruct_user_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --sequence_parallel_size 1 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_top_p 1.0 \
    --vllm_enable_prefix_caching \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 250 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub False
