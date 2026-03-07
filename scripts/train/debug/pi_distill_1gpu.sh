#!/bin/bash
# pi-Distill 1-GPU debug script (no mason, runs directly)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1

uv run python open_instruct/grpo_fast.py \
    --exp_name pi_distill_1gpu_debug \
    --pi_distill \
    --pi_distill_alpha 0.5 \
    --beta 1.0 \
    --load_ref_policy false \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --temperature 0.7 \
    --learning_rate 1e-6 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --seed 1 \
    --local_eval_every 2 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --active_sampling \
    --async_steps 8 \
    --loss_fn dapo \
    --clip_higher 0.28 \
    --advantage_normalization_type centered
