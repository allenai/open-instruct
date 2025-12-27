#!/bin/bash


BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
export exp_name=test_olmo3_32b_rl_run_${RANDOM}
export data_mix="hamishivi/math_rlvr_mixture_dpo 1.0 hamishivi/code_rlvr_mixture_dpo 1.0 hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter 30186 allenai/rlvr_general_mix-keyword-filtered 21387"
export model_path=/weka/oe-adapt-default/hamishi/model_checkpoints/olmo3-merge-32b-1e-4-5e-5/olmo3-merge-32b-1e-4-5e-5/


uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/augusta \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --gs_model_name "sft_olmo3_32b_rl_run_testing" \
    --preemptible \
    --num_nodes 28 \
    --gpus 8 \
    --max_retries 0 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    --env HOSTED_VLLM_API_BASE=http://ceres-cs-aus-447.reviz.ai2.in:8001/v1 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 2e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list ${data_mix} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/omega-combined 8 allenai/IF_multi_constraints_upto5 8 saurabh5/rlvr_acecoder_filtered 8 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path ${model_path} \
        --chat_template_name olmo_thinker \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 8 8 8 8 8 8 8 8 8 8 \
        --vllm_num_engines 6 \
        --gather_whole_model False \
        --vllm_tensor_parallel_size 8 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 50 \
        --save_freq 25 \
        --eval_priority urgent \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --clip_higher 0.272 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 100 \
        --backend_timeout 1200 \
        --inflight_updates true \
        --async_steps 8 \
        --active_sampling \
        --advantage_normalization_type centered \
        --truncated_importance_sampling_ratio_cap 2.0 \
        --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --oe_eval_tasks mmlu:cot::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,gpqa:0shot_cot::qwen3-instruct,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek \
        --vllm_enforce_eager \
        --deepspeed_zpg 32
