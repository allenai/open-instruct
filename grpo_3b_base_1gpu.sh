#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH -p main

source mila.sh
exp_name="grpo_qwen3b_base"

uv run open_instruct/grpo_tinyzero.py \
    --exp_name $exp_name \
    --output_dir $SCRATCH/open_instruct/results/ \
    --dataset_name Jiayi-Pan/Countdown-Tasks-3to4 \
    --dataset_train_split train[1000:] \
    --dataset_eval_split train[:1000] \
    --max_token_length 256 \
    --max_prompt_token_length 256 \
    --response_length 1024 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 4 \
    --total_episodes 64000 \
    --model_name_or_path Qwen/Qwen2.5-3B \
    --base_prompt \
    --stop_strings "<|endoftext|>" "User:" "</answer>" "Assistant:" \
    --apply_r1_style_format_reward \
    --r1_style_format_reward 0.1 \
    --apply_verifiable_reward \
    --verification_reward 0.9 \
    --non_stop_penalty False \
    --temperature 0.7 \
    --learning_rate 5e-7 \
    --num_epochs 1 \
    --beta 0.001 \
    --seed 3 \
    --num_evals 5 \
    --save_freq 50 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.55 \
    --vllm_enable_prefix_caching \
    --deepspeed_stage 2 \
    --single_gpu_mode \
    --async_mode False \
    --liger_kernel \
    --vllm_sync_backend gloo \
    --gradient_checkpointing \
    --vllm_sleep_level 1 \
    --offload_ref \
    --fused_optimizer \
    --with_tracking $@

    # --vllm_enforce_eager \
    # --vllm_enable_prefix_caching \
    # --dataset_mixer_list_splits train \
    # --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    # --dataset_mixer_eval_list_splits train \
    # --dataset_local_cache_dir $SCRATCH/open_instruct/local_dataset_cache \
