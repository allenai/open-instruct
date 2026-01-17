export CRAWL4AI_API_URL=http://w046sy3a7g.execute-api.us-east-1.amazonaws.com/prod
export CRAWL4AI_API_KEY=$(beaker secret read hamishivi_CRAWL4AI_API_KEY --workspace ai2/olmo-instruct)
export SERPER_API_KEY=$(beaker secret read hamishivi_SERPER_API_KEY --workspace ai2/olmo-instruct)
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python open_instruct/grpo_fast.py \
        --exp_name grpo_fast_debug_tool_run \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 4 \
        --num_unique_prompts_rollout 8 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir output \
        --kl_estimator 2 \
        --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 8 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 2048 \
        --pack_length 8192 \
        --model_name_or_path Qwen/Qwen3-0.6B \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 2 \
        --sequence_parallel_size 2 \
        --vllm_num_engines 1 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 50 \
        --try_launch_beaker_eval_jobs_on_weka False \
        --gradient_checkpointing \
        --backend_timeout 1200 \
        --inflight_updates false \
        --async_steps 1 \
        --advantage_normalization_type centered \
        --truncated_importance_sampling_ratio_cap 2.0 \
        --system_prompt_override_file open_instruct/tools/system_prompts/olmo3_serper_crawl_python_prompt.txt \
        --tools search_serper browse_crawl4ai code \
        --code_tool_api_endpoint https://open-instruct-tool-server-10554368204.us-central1.run.app/execute \
        # --single_gpu_mode \
        # --vllm_sync_backend gloo \
        # --vllm_gpu_memory_utilization 0.3 \