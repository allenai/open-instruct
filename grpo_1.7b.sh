#!/bin/bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH -p main

source mila.sh
exp_name="grpo_qwen3_1.7b_base"

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
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --total_episodes 64000 \
    --model_name_or_path Qwen/Qwen3-1.7B-Base \
    --base_prompt \
    --stop_strings "<|endoftext|>" "</answer>" \
    --apply_r1_style_format_reward \
    --r1_style_format_reward 0.1 \
    --apply_verifiable_reward \
    --verification_reward 0.9 \
    --non_stop_penalty False \
    --temperature 1.0 \
    --eval_temperature 0.7 \
    --top_p 0.95 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching \
    --beta 0.001 \
    --seed 3 \
    --num_evals 4 \
    --save_freq 1000 \
    --vllm_gpu_memory_utilization 0.55 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --async_mode False \
    --vllm_sync_backend gloo \
    --offload_ref \
    --vllm_sleep_level 1 \
    --fused_optimizer \
    --with_tracking $@
    #

    # --liger_kernel \
    # --vllm_enforce_eager \
    # --gradient_checkpointing \
    # --dataset_mixer_list_splits train \
    # --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    # --dataset_mixer_eval_list_splits train \
    # --dataset_local_cache_dir $SCRATCH/open_instruct/local_dataset_cache \
