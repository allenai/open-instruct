export mixin_it_up="saurabh5/rlvr_acecoder_filtered 19000 saurabh5/open-code-reasoning-rlvr-stdio 19000 saurabh5/klear-code-rlvr 11000 saurabh5/synthetic2-rlvr-code-compressed 5000 hamishivi/omega-combined 20000 hamishivi/polaris_53k 18000 TTTXXX01/MathSub-30K 9000 hamishivi/DAPO-Math-17k-Processed 7000 allenai/IF_multi_constraints_upto5 36000 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 22000 hamishivi/virtuoussy_multi_subject_rlvr 14000"

# testing general-ish data with from base + llm judge
for split_var in mixin_it_up; do
    split_value="${!split_var}"
    exp_name=1808rl_olmo_sft_mix_testinflight_${split_var}
    exp_name="${exp_name}_${RANDOM}"

    uv run mason.py \
        --cluster ai2/augusta-google-1 --image hamishivi/open_instruct_batching \
        --pure_docker_mode \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --preemptible \
        --num_nodes 4 \
        --max_retries 0 \
        --env RAY_CGRAPH_get_timeout=300 \
        --gs_model_name olmo2_7b_sft_lc_ot3_full_regen_wc_oasst_ccn_pif_qif_wgwj_syn2_aya_tgpt_ncode_scode \
        --env VLLM_DISABLE_COMPILE_CACHE=1 \
        --env HOSTED_VLLM_API_BASE=http://saturn-cs-aus-234.reviz.ai2.in:8001/v1 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env LITELLM_LOG="ERROR" \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 16 \
        --num_mini_batches 4 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator kl3 \
        --inflight_updates true \
        --dataset_mixer_list ${split_value} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/omega-combined 8 allenai/IF_multi_constraints_upto5 8 saurabh5/rlvr_acecoder_filtered 8 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode \
	--tokenizer_name_or_path allenai/olmo-2-1124-7b \
        --chat_template_name olmo_thinker \
        --stop_strings "</answer>" \
        --non_stop_penalty True \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 12 \
        --vllm_tensor_parallel_size 2 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 100 \
        --eval_priority high \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --clip_higher 0.272 \
        --allow_world_padding False \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --oe_eval_max_length 32768 \
        --verbose \
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker,alpaca_eval_v3::hamish_zs_reasoning,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,omega_500:0-shot-chat"
done
