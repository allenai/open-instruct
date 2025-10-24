# dataset_mix="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 1.0 saurabh5/MATH_3000_Filtered_olmo_completions_new_template_filtered 1.0"
dataset_mix="hamishivi/rlvr_acecoder_filtered_filtered 33333 hamishivi/IF_multi_constraints_upto5_filtered 33333 hamishivi/rlvr_orz_math_57k_collected_filtered 33334"
# HF_CACHE_ROOT=${HF_CACHE_ROOT:-/filestore/.cache/huggingface/math_and_code}
# mkdir -p "${HF_CACHE_ROOT}/datasets" "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/transformers" 2>/dev/null || true
evals="gsm8k::zs_cot_latex_deepseek,minerva_math::hamish_zs_reasoning_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,ifeval::hamish_zs_reasoning_deepseek"


# # ones I'll use:
# # code:
# hamishivi/rlvr_acecoder_filtered_filtered 33333 # 62,814
# # if:
# hamishivi/IF_multi_constraints_upto5_filtered 33333 # 95,279
# # math:
# hamishivi/rlvr_orz_math_57k_collected_filtered 33334 # 56,250

# # code:
# hamishivi/rlvr_acecoder_filtered_filtered 20000 # 62,814
# # if:
# hamishivi/IF_multi_constraints_upto5_filtered 38000 # 95,279
# # math:
# hamishivi/omega-combined-no-boxed_filtered 20000 # 62,841
# hamishivi/rlvr_orz_math_57k_collected_filtered 14000 # 56,250
# hamishivi/polaris_53k 14000 # 53,291
# hamishivi/MathSub-30K_filtered 9000 # 29,254
# hamishivi/DAPO-Math-17k-Processed_filtered 7000 # 12,643
# # general:
# hamishivi/new-wildchat-english-general_filtered 19000 # 19,403
# hamishivi/virtuoussy_multi_subject_rlvr_filtered 20000 # 572,431
# hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered 22000 # 43,382

model_name_or_path="/weka/oe-training-default/ai2-llm/checkpoints/tylerr/long-context/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_packing_yarn-fullonly_50B-fb13a737/step11921-hf"
gs_model_name="olmo2.5-final-long-context"

exp_name="test_exp"
EXP_NAME=${EXP_NAME:-${exp_name}}
# NUM_GPUS=${NUM_GPUS:-8}

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image jacobm/open_instruct_dev_random_rewards2 \
    --preemptible \
    --num_nodes 3 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gs_model_name $gs_model_name \
    --no_auto_dataset_cache \
    --pure_docker_mode \
    --gpus 8 \
    --budget ai2/oe-adapt -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list ${dataset_mix} \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list hamishivi/rlvr_acecoder_filtered_filtered 4 hamishivi/IF_multi_constraints_upto5_filtered 4 hamishivi/rlvr_orz_math_57k_collected_filtered 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path ${model_name_or_path} \
    --chat_template_name olmo \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 10000000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 16 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --random_rewards true \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --keep_last_n_checkpoints -1 \
    --mask_truncated_completions True \
    --async_steps 4 \
    --inflight_updates \
    --oe_eval_max_length 16384 \
    --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program\
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks ${evals} \
    --eval_on_step_0 False \
    --output_dir /weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline \
    --checkpoint_state_dir /weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_states \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto $@

    # --truncated_importance_sampling_ratio_cap 2.0 \