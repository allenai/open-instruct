export omega_easy="nouhad/omega-easy 56878"
export omega_medium="nouhad/omega-medium 56878"
export omega_hard="nouhad/omega-hard 56878"
export code_with_omega="saurabh5/llama-nemotron-rlvr 1.0 nouhad/omega-all-combined 1.0"
export code_with_omega="saurabh5/llama-nemotron-rlvr 1.0 nouhad/omega-all-combined 1.0"

for model in qwen2_5_oth2; do
for split_var in code_with_omega; do
for minibatches in 1; do
    split_value="${!split_var}"
    exp_name=code-omega-rlep-${model}_${split_var}  # Added "rlep" to distinguish
    if ["$model" == "olmo2-lc-OTH3-jm"]; then
        model_name_or_path=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/rl-sft/olmo2-7B-FINAL-lc-OT3-full-regen-wc-oasst-ccn-pif-qif-wgwj-syn2-aya-tgpt-ncode-scode
        chat_template_name=tulu_thinker
        add_bos=True
    elif [ "$model" == "qwen3" ]; then
        model_name_or_path=hamishivi/qwen3_openthoughts2
        chat_template_name=tulu_thinker
        add_bos=False
    elif [ "$model" == "olmo2_lc" ]; then
        model_name_or_path=hamishivi/olmo2_lc_ot2
        chat_template_name=tulu_thinker
        add_bos=True
    elif [ "$model" == "qwen2_5_oth2" ]; then
        model_name_or_path=hamishivi/qwen2_5_openthoughts2
        chat_template_name=tulu_thinker
        add_bos=False
    elif [ "$model" == "qwen2_5" ]; then
        model_name_or_path=Qwen/Qwen2.5-7B
        chat_template_name=tulu_thinker
        add_bos=False
    elif [ "$model" == "qwen2_5_7b_math" ]; then
        model_name_or_path=Qwen/Qwen2.5-Math-7B
        chat_template_name=tulu_thinker
        add_bos=False
    fi
    python mason.py \
        --cluster ai2/jupiter  \
        --pure_docker_mode \
        --workspace ai2/olmo-instruct \
        --gs_model_name "olmo2-omega-rlep" \
        --priority urgent \
        --preemptible \
        --num_nodes 5 \
        --max_retries 0 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast_code.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 128 \
        --num_mini_batches ${minibatches} \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${split_value} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list saurabh5/llama-nemotron-rlvr 32 \
        --dataset_mixer_eval_list_splits train \
        --dataset_skip_cache true \
        --add_bos ${add_bos} \
        --max_token_length 10240 \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 20480 \
        --model_name_or_path ${model_name_or_path} \
        --chat_template_name ${chat_template_name} \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 8 \
        --vllm_num_engines 16 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --save_freq 50 \
        --local_eval_every 1953 \
        --eval_priority high \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --clip_higher 0.28 \
        --oe_eval_max_length 32768 \
        --allow_world_padding True \
        --enable_sqlite_logging True \
        --enable_experience_replay True \
        --replay_min_success_count 10 \
        --replay_per_group_limit 3 \
        --replay_per_query_limit 5 \
        --replay_sampling_strategy recent \
        --replay_max_age_steps 500 \
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker,omega:0-shot-chat"
done
done
done