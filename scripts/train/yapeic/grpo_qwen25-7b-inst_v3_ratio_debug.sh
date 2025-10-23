#!/usr/bin/env bash
set -euo pipefail

# Hosted judge API base (override if your judge is elsewhere)
export HOSTED_VLLM_API_BASE=http://saturn-cs-aus-241.reviz.ai2.in:8001/v1
export CUDA_VISIBLE_DEVICES=0,1

python open_instruct/grpo_fast.py \
    --exp_name grpo_qwen25-3b-inst_v3_ratio_debug \
    --beta 0.0 \
    --num_unique_prompts_rollout 1 \
    --num_samples_per_prompt_rollout 2 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_ratio_with_format_10k.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_ratio_with_format_10k_test.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --stop_strings "</answer>" \
    --chat_template_name r1_simple_chat_postpend_think \
    --apply_verifiable_reward True \
    --llm_judge_model "hosted_vllm/yapeichang/distill_judge_qwen3-8b_sft_v2_fixed_data" \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --total_episodes 200 \
    --deepspeed_stage 0 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 1 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 1 \
    --checkpoint_state_freq -1 \
    --save_freq -1