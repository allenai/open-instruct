export split_int_mix_3="hamishivi/omega-combined 63033 allenai/IF_multi_constraints_upto5 63033 saurabh5/rlvr_acecoder_filtered 63033 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 63033"

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"
# testing general-ish data with from base + llm judge
for split_var in split_int_mix_3; do
    split_value="${!split_var}"
    exp_name=2507rl_qwen2ot2_sft_mix_${split_var}
    exp_name="${exp_name}_${RANDOM}"

    uv run python mason.py \
        --cluster ai2/augusta \
        --image "$BEAKER_IMAGE" \
        --pure_docker_mode \
        --workspace ai2/olmo-instruct \
        --priority high \
        --preemptible \
        --num_nodes 2 \
        --max_retries 0 \
        # torch compile caching seems consistently broken, but the actual compiling isn't.
        # Not sure why, for now we have disabled the caching (VLLM_DISABLE_COMPILE_CACHE=1).
        --env VLLM_DISABLE_COMPILE_CACHE=1 \
        --env HOSTED_VLLM_API_BASE=http://saturn-cs-aus-253.reviz.ai2.in:8001/v1 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env LITELLM_LOG="ERROR" \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 32 \
        --num_mini_batches 4 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list "${split_value}" \
        --dataset_mixer_list_splits "train" \
        --dataset_mixer_eval_list "hamishivi/omega-combined 8 allenai/IF_multi_constraints_upto5 8 saurabh5/rlvr_acecoder_filtered 8 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4" \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 8192 \
        --pack_length 12384 \
        --model_name_or_path ai2-adapt-dev/tulu_3_long_finetune_qwen_7b_reg \
        --chat_template_name tulu_thinker \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10_000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 100 \
        --save_freq 100 \
        --eval_priority high \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 131072 \
        --clip_higher 0.272 \
        --oe_eval_max_length 32768 \
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker"
done
