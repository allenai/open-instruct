#!/bin/bash
exp_name="grpo_qwen3_1.7b_base"

HF_HOME=/weka/oe-adapt-default/allennlp/.cache/huggingface

uv run open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --output_dir tmp \
    --dataset_mixer_list mnoukhov/rlvr_countdown 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list mnoukhov/rlvr_countdown 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 256 \
    --max_prompt_token_length 256 \
    --response_length 1024 \
    --pack_length 2048 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 8 \
    --total_episodes 64000 \
    --model_name_or_path Qwen/Qwen3-1.7B-Base \
    --stop_strings "<|endoftext|>" "</answer>" \
    --chat_template_name r1_simple_chat_postpend_think \
    --apply_r1_style_format_reward \
    --additive_format_reward \
    --r1_style_format_reward 0.1 \
    --apply_verifiable_reward \
    --verification_reward 0.9 \
    --non_stop_penalty False \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching \
    --beta 0.001 \
    --seed 3 \
    --num_evals 4 \
    --save_freq 1000 \
    --vllm_gpu_memory_utilization 0.5 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --async_mode False \
    --gradient_checkpointing \
    --vllm_sync_backend gloo \
    --offload_ref \
    --vllm_sleep_mode \
    --fused_optimizer \
    --wandb_project_name r1 \
    --wandb_entity mila-language-drift \
    --with_tracking False $@
    #

    # --eval_temperature 0.7 \
    # --base_prompt \
    # --liger_kernel \
    # --vllm_enforce_eager \
    # --dataset_mixer_list_splits train \
    # --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    # --dataset_mixer_eval_list_splits train \
    # --dataset_local_cache_dir $SCRATCH/open_instruct/local_dataset_cache \
