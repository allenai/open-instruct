exp_name="base_smollm_grpo_${RANDOM}"
uv run open_instruct/grpo_tinyzero.py \
    --exp_name $exp_name \
    --output_dir output/$exp_name \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 256 \
    --max_prompt_token_length 256 \
    --response_length 1024 \
    --pack_length 8192 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --non_stop_penalty False \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 128000 \
    --per_device_train_batch_size 1 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.001 \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 5 \
    --save_freq 50 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enforce_eager \
    --single_gpu_mode $@

    # --deepspeed_stage 2 \
    # --gradient_checkpointing \
