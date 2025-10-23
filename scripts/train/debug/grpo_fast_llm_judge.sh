# note: judge may not be alive, internal ai2 host.
export HOSTED_VLLM_API_BASE=http://saturn-cs-aus-234.reviz.ai2.in:8001/v1

uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list hamishivi/virtuoussy_multi_subject_rlvr 64 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/virtuoussy_multi_subject_rlvr 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 8 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 200 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
    --llm_judge_timeout 600 \
    --llm_judge_max_tokens 2048 \
    --llm_judge_max_context_length 32768 \
    # --with_tracking
