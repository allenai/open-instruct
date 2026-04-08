model_onpolicy=hamishivi/1810_rl_rag_NAR8_onpolicy_step300
gpt5_sft=hamishivi/2010_rl_rag_NAR8_testing64_gpt5_sft_step300
gpt5_under_trained=hamishivi/2010_rl_rag_NAR8_testing64_gpt5_under_trained_step300
base_model=hamishivi/2010_rl_rag_NAR8_testing64_base_model_step300

for model in model_onpolicy gpt5_sft gpt5_under_trained base_model; do
    exp_name="2910_get_rubrics_${model}_${RANDOM}"
    model_path=${!model}
    python mason.py \
        --cluster ai2/jupiter \
        --image hamishivi/open_instruct_rl_rag_testing3 \
        --pure_docker_mode \
        --budget ai2/oe-adapt \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --preemptible \
        --num_nodes 2 \
        --gpus 8 \
        --max_retries 0 \
        --secret S2_API_KEY=hamishivi_S2_API_KEY \
        --secret SERPER_API_KEY=hamishivi_SERPER_API_KEY \
        --env CRAWL4AI_BLOCKLIST_PATH=/stage/rl-rag-mcp/utils/crawl4ai_block_list.txt \
        --env MASSIVE_DS_URL='http://saturn-cs-aus-232.reviz.ai2.in:44177/search' \
        --env MCP_MAX_CONCURRENT_CALLS=512 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env RUBRIC_JUDGE_MODEL=gpt-4.1-mini \
        --env LITELLM_MAX_CONCURRENT_CALLS=32 \
        --env LITELLM_DEFAULT_TIMEOUT=600 \
        --env LITELLM_LOG=DEBUG \
        -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
            --exp_name ${exp_name} \
            --wandb_project_name rl-rag \
            --beta 0.001 \
            --num_samples_per_prompt_rollout 8 \
            --num_unique_prompts_rollout 64 \
            --num_mini_batches 1 \
            --num_epochs 1 \
            --learning_rate 5e-7 \
            --per_device_train_batch_size 1 \
            --output_dir /output \
            --kl_estimator kl3 \
            --dataset_mixer_list rl-rag/rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics_adaptive_rubric 1.0 rl-rag/rl_rag_train_os_0915_2k_search_rubric_longform_rubrics_adaptive_rubric 1.0 rl-rag/rl_rag_train_sa_3k_longform_rubrics_adaptive_rubric 1.0 rl-rag/RaR-Medicine-20k-o3-mini-converted 3000 rl-rag/RaR-Science-20k-o3-mini-converted 1000 \
            --dataset_mixer_list_splits train \
            --dataset_mixer_eval_list rl-rag/healthbench_all_adaptive_rubric 16 \
            --dataset_mixer_eval_list_splits test \
            --apply_adaptive_rubric_reward true \
            --normalize_rubric_scores false \
            --use_rubric_buffer true \
            --use_static_rubrics_as_persistent_rubrics true \
            --max_active_rubrics 5 \
            --max_token_length 10240 \
            --max_prompt_token_length 2048 \
            --response_length 16384 \
            --pack_length 18500 \
            --model_name_or_path ${model_path} \
            --non_stop_penalty False \
            --non_stop_penalty_value 0.0 \
            --temperature 1.0 \
            --ground_truths_key ground_truth \
            --sft_messages_key messages \
            --total_episodes 3072 \
            --deepspeed_stage 3 \
            --num_learners_per_node 8 \
            --vllm_num_engines 8 \
            --vllm_tensor_parallel_size 1 \
            --lr_scheduler_type constant \
            --apply_verifiable_reward true \
            --seed 1 \
            --num_evals 500 \
            --save_freq 50 \
            --try_launch_beaker_eval_jobs_on_weka False \
            --gradient_checkpointing \
            --with_tracking \
            --max_tool_calls 10 \
            --only_reward_good_outputs False \
            --tools mcp \
            --checkpoint_state_freq 50 \
            --save_adaptive_rubrics true \
            --save_traces true \
            --mcp_parser_name v20250824 \
            --system_prompt_file open_instruct/tools/system_prompts/unified_tool_calling_v20250907.yaml  \
            --mcp_tool_names 'snippet_search,google_search,browse_webpage' \
            --mcp_server_command "'python -m rl-rag-mcp.mcp_agents.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp'"
done