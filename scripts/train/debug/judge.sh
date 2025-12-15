export HOSTED_VLLM_API_BASE=http://saturn-cs-aus-230.reviz.ai2.in:8001/v1

# new version
python open_instruct/grpo_fast.py \
    --dataset_mixer_list faezeb/tulu_3_rewritten_100k-no-math 20000 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/tulu_3_rewritten_100k 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 4096 \
    --max_prompt_token_length 2048 \
    --response_length 512 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --stop_strings "</answer>" \
    --kl_estimator 2 \
    --apply_verifiable_reward true \
    --apply_r1_style_format_reward true \
    --non_stop_penalty False \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu_thinker_r1_style \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant \
    --total_episodes 2048 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.0 \
    --seed 3 \
    --local_eval_every 1 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --llm_judge_model "hosted_vllm/Qwen/Qwen3-32B" \
    --llm_judge_timeout 600 \
    --llm_judge_max_tokens 1024 \
    --llm_judge_max_context_length 8192

# 8192
# initial saurabh version
# python open_instruct/grpo_fast.py \
#     --dataset_mixer_list faezeb/tulu_3_rewritten_100k-no-math 512 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list ai2-adapt-dev/general-thoughts-100k-rewritten-v2-ifeval 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 4096 \
#     --max_prompt_token_length 2048 \
#     --response_length 512 \
#     --pack_length 4096 \
#     --per_device_train_batch_size 1 \
#     --num_unique_prompts_rollout 64 \
#     --num_samples_per_prompt_rollout 16 \
#     --model_name_or_path Qwen/Qwen2.5-0.5B \
#     --stop_strings "</answer>" \
#     --apply_verifiable_reward true \
#     --apply_r1_style_format_reward true \
#     --non_stop_penalty True \
#     --non_stop_penalty_value 0.0 \
#     --temperature 0.7 \
#     --ground_truths_key ground_truth \
#     --chat_template_name tulu_thinker_r1_style \
#     --learning_rate 3e-7 \
#     --total_episodes 2048 \
#     --deepspeed_stage 2 \
#     --num_epochs 1 \
#     --num_learners_per_node 1 \
#     --vllm_tensor_parallel_size 1 \
#     --beta 0.01 \
#     --seed 3 \
#     --local_eval_every 1 \
#     --vllm_sync_backend gloo \
#     --vllm_gpu_memory_utilization 0.5 \
#     --save_traces \
#     --vllm_enforce_eager \
#     --gradient_checkpointing \
#     --single_gpu_mode \
#     --push_to_hub false \
#     --llm_judge_model "azure/gpt-4.1-standard"
