#!/bin/bash
# Usage: git checkout $SOME_COMMIT_HASH && ./scripts/train/build_image_and_launch_dirty.sh scripts/train/olmo3/benchmark_infra.sh
set -euo pipefail

IMAGE_NAME="${1:-hamishivi/olmo_25_image_latest3}"

export mixin_it_up="hamishivi/math_rlvr_mixture_dpo 1.0 hamishivi/code_rlvr_mixture_dpo 1.0 hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter 30186 hamishivi/rlvr_general_mix 21387"
#export MODEL_NAME="/weka/oe-adapt-default/hamishi/model_checkpoints/olmo2.5-6T-LC_R1-reasoning_mix_1_with_yarn_olmo3_ver/"
export MODEL_NAME="hamishivi/qwen2_5_openthoughts2"


for split_var in mixin_it_up; do
    split_value="${!split_var}"
    exp_name=olmo3_dpo_rl_final_mix_jupiter_${RANDOM}
    exp_name="${exp_name}_${RANDOM}"

    uv run mason.py \
        --description "Ablating infrastructure improvements." \
        --cluster ai2/augusta --image "$IMAGE_NAME" \
        --pure_docker_mode \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --preemptible \
        --num_nodes 2 \
        --max_retries 0 \
	--timeout 2h \
        --env RAY_CGRAPH_get_timeout=300 \
        --env TORCH_NCCL_ENABLE_MONITORING=0 \
        --gs_model_name sm0922-rsn-dpo-delta-yolo_scottmix1_150k-8e-8__42__1758585338-olmo-25-final2 \
        --env VLLM_DISABLE_COMPILE_CACHE=1 \
        --env HOSTED_VLLM_API_BASE=http://saturn-cs-aus-240.reviz.ai2.in:8001/v1 \
        --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        --env LITELLM_LOG="ERROR" \
        --budget ai2/oe-adapt \
        --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
	--async_steps=4 \
        --num_samples_per_prompt_rollout 2 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /output \
        --kl_estimator 2 \
        --dataset_mixer_list ${split_value} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list hamishivi/omega-combined 8 allenai/IF_multi_constraints_upto5 8 saurabh5/rlvr_acecoder_filtered 8 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path "$MODEL_NAME" \
        --chat_template_name olmo_thinker \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
	--with_tracking \
        --local_eval_every 50 \
        --gradient_checkpointing \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --clip_higher 0.2 \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 10000 \
        --backend_timeout 1200 \
        --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto \
        --oe_eval_tasks "mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek"
done
