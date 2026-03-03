#!/bin/bash
# Tool use basic with rl, starting from SFT.

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')

export SERPER_API_KEY=$(beaker secret read ${BEAKER_USER}_SERPER_API_KEY --workspace ai2/open-instruct-dev)
export JINA_API_KEY=$(beaker secret read ${BEAKER_USER}_JINA_API_KEY --workspace ai2/open-instruct-dev)
export S2_API_KEY=$(beaker secret read ${BEAKER_USER}_S2_API_KEY --workspace ai2/open-instruct-dev)

uv run python mason.py \
       --cluster ai2/jupiter \
       --image hamishivi/open_instruct_dev_3101 \
       --description "OLMo-3 multinode tool use test" \
       --pure_docker_mode \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --preemptible \
       --num_nodes 2 \
       --max_retries 0 \
       --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --budget ai2/oe-adapt \
       --secret SERPER_API_KEY=${BEAKER_USER}_SERPER_API_KEY \
       --secret JINA_API_KEY=${BEAKER_USER}_JINA_API_KEY \
       --secret S2_API_KEY=${BEAKER_USER}_S2_API_KEY \
       --env HOSTED_VLLM_API_BASE=http://saturn-cs-aus-241.reviz.ai2.in:8001/v1 \
       --gpus 8 \
       -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
   --dataset_mixer_list hamishivi/rlvr_acecoder_filtered_filtered 20000 hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000 hamishivi/rl_rag_shortformqa 1.0 \
   --dataset_mixer_list_splits train \
   --dataset_mixer_eval_list hamishivi/rlvr_acecoder_filtered_filtered 2 hamishivi/omega-combined-no-boxed_filtered 2 hamishivi/rlvr_orz_math_57k_collected_filtered 2 hamishivi/polaris_53k 2 hamishivi/MathSub-30K_filtered 2 hamishivi/DAPO-Math-17k-Processed_filtered 2 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 2 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 2 hamishivi/rl_rag_shortformqa 2 \
   --dataset_mixer_eval_list_splits train \
   --max_prompt_token_length 2048 \
   --response_length 63488 \
   --pack_length 65536 \
   --inflight_updates True \
   --per_device_train_batch_size 1 \
   --num_unique_prompts_rollout 32 \
   --num_samples_per_prompt_rollout 8 \
   --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
   --apply_verifiable_reward true \
   --temperature 1.0 \
   --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
   --llm_judge_timeout 600 \
   --llm_judge_max_tokens 2048 \
   --llm_judge_max_context_length 32768 \
   --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
   --code_pass_rate_reward_threshold 0.99 \
   --ground_truths_key ground_truth \
   --sft_messages_key messages \
   --exp_name olmo3_7b_tool_use_test \
   --learning_rate 5e-7 \
   --total_episodes 100000000000 \
   --deepspeed_stage 3 \
   --sequence_parallel_size 4 \
   --with_tracking \
   --num_epochs 1 \
   --num_learners_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --beta 0.0 \
   --seed 1 \
   --local_eval_every 10 \
   --gradient_checkpointing \
   --push_to_hub false \
   --output_dir /output \
   --kl_estimator 2 \
   --non_stop_penalty False \
   --num_mini_batches 1 \
   --lr_scheduler_type constant \
   --save_freq 100 \
   --try_launch_beaker_eval_jobs_on_weka False \
   --max_tool_calls 5 \
   --vllm_enable_prefix_caching \
   --no_resampling_pass_rate 0.875 \
   --advantage_normalization_type centered \
   --async_steps 8 \
   --active_sampling \
   --truncated_importance_sampling_ratio_cap 2.0 \
   --inflight_updates \
   --backend_timeout 1200

