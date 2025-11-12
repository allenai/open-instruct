# note: judge may not be alive, internal ai2 host.
export HOSTED_VLLM_API_BASE=http://saturn-cs-aus-233.reviz.ai2.in:8001/v1

uv run python open_instruct/grpo_fast.py \
    --beta 0.0 \
    --num_unique_prompts_rollout 1 \
    --num_samples_per_prompt_rollout 1 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_binary_with_format_10k.jsonl 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list /weka/oe-training-default/yapeic/proc-data/data/dclm/tutorial_subset/grpo_data/v3_goal_resource_steps_low_variation_simple_template_binary_with_format_10k_test.jsonl 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 4096 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --pack_length 4096 \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --apply_verifiable_reward True \
    --llm_judge_model "hosted_vllm/yapeichang/distill_judge_qwen3-8b_sft_v2_fixed_data" \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --total_episodes 160000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 1 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 1 \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 80 \
    --save_freq 250 \
    --keep_last_n_checkpoints 10 \
    --gradient_checkpointing \
