BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}

# data mixes
nonreasoner_integration_mix_decon="hamishivi/rlvr_acecoder_filtered_filtered 20000 hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000" #hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered 22000 hamishivi/virtuoussy_multi_subject_rlvr_filtered 20000 hamishivi/new-wildchat-english-general_filtered 19000"

# eval suite
general_evals_int="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"

# model checkpoint
model_name_or_path="/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newb-tpc-d5-lbc100-bal-1e-6-1__42__1763293644" # nov 16 tentative final checkpoint row 33 # was replace by row 27

# cluster
cluster=ai2/augusta
#template
chat_template=olmo123 #olmo

NUM_GPUS=${NUM_GPUS:-8}
hosted_vllm=""
exp_name="grpo_math_only_p64_4_8k_${gs_model_name}"
gs_model_name="olmo3-instruct-dpo-hpz1"

EXP_NAME=${EXP_NAME:-${exp_name}}

uv run python mason.py \
        --description $exp_name \
        --task_name ${EXP_NAME} \
        --cluster ${cluster} \
        --workspace ai2/olmo-instruct  \
        --priority urgent \
        --pure_docker_mode \
        --image $BEAKER_IMAGE \
        --preemptible \
        --num_nodes 8 \
        --max_retries 5 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env HOSTED_VLLM_API_BASE=$hosted_vllm \
        --gs_model_name $gs_model_name \
        --gpus ${NUM_GPUS} \
        --budget ai2/oe-adapt -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${EXP_NAME} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 4 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --kl_estimator 2 \
        --dataset_mixer_list ${nonreasoner_math_mix_decon} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/omega-combined 4 allenai/IF_multi_constraints_upto5 4 saurabh5/rlvr_acecoder_filtered 4 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 --response_length 8192 --pack_length 11264 \
        --model_name_or_path ${model_name_or_path} \
        --chat_template_name ${chat_template} \
        --stop_strings "</answer>" \
        --non_stop_penalty False \
        --temperature 1.0 \
        --total_episodes 1024000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 56 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 50 \
        --checkpoint_state_freq 50 \
        --beaker_eval_freq 50 \
        --gradient_checkpointing \
        --with_tracking \
        --vllm_enable_prefix_caching \
        --clip_higher 0.272 \
        --mask_truncated_completions False \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --oe_eval_max_length 32768 \
        --try_launch_beaker_eval_jobs_on_weka True \
        --oe_eval_tasks ${general_evals_int} \
        --eval_priority urgent \
        --code_pass_rate_reward_threshold 0.99 \
        --inflight_updates true \
        --async_steps 8 \
        --active_sampling \
        --advantage_normalization_type centered \
        --no_resampling_pass_rate 0.875 \
