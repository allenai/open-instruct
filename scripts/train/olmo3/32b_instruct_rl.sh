#!/bin/bash
# Wandb: https://wandb.ai/ai2-llm/open_instruct_internal/runs/9gzo45lx
# Beaker: https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KAJQH1X2PRZP5VYZ1F0Z96KK
# Commit: 8d8232cc

BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
export exp_name=jacobm_olmo3_instruct_32b_rl_2e-6-jupiter-16k
export data_mix="hamishivi/rlvr_acecoder_filtered_filtered 20000 hamishivi/omega-combined-no-boxed_filtered 20000 hamishivi/rlvr_orz_math_57k_collected_filtered 14000 hamishivi/polaris_53k 14000 hamishivi/MathSub-30K_filtered 9000 hamishivi/DAPO-Math-17k-Processed_filtered 7000 allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered 38000 allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered 50000"
export model_path=/weka/oe-adapt-default/allennlp/deletable_checkpoint/jacobm/olmo3-32b-DPO-1116-match-7b-1e-6__42__1763403219
export DATASET_MIXER_EVAL_LIST="hamishivi/omega-combined 4 allenai/IF_multi_constraints_upto5 4 saurabh5/rlvr_acecoder_filtered 4 hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2 4 hamishivi/virtuoussy_multi_subject_rlvr 4"
export OE_EVAL_TASKS="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek"

uv run python mason.py \
    --budget ai2/oe-adapt \
    --cluster ai2/jupiter-cirrascale-2 \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --preemptible \
    --num_nodes 12 \
    --gpus 8 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 2e-6 \
        --per_device_train_batch_size 1 \
        --kl_estimator kl3 \
        --dataset_mixer_list ${data_mix} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list ${DATASET_MIXER_EVAL_LIST} \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 16384 \
        --pack_length 19456 \
        --model_name_or_path ${model_path} \
        --chat_template_name olmo123 \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 8 8 \
        --vllm_num_engines 16 \
        --inference_batch_size 200 \
        --gather_whole_model False \
        --vllm_tensor_parallel_size 4 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 543210 \
        --local_eval_every 50 \
        --save_freq 50 \
        --beaker_eval_freq 50 \
        --eval_priority urgent \
        --try_launch_beaker_eval_jobs_on_weka True \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --clip_higher 0.272 \
        --allow_world_padding False \
        --use_fp8_kv_cache False \
        --code_api_url https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --code_max_execution_time 6 \
        --oe_eval_max_length 32768 \
        --checkpoint_state_freq 100 \
        --backend_timeout 1200 \
        --inflight_updates true \
        --async_steps 8 \
        --active_sampling \
        --advantage_normalization_type centered \
        --truncated_importance_sampling_ratio_cap 2.0 \
        --oe_eval_tasks "${OE_EVAL_TASKS}" \
        --oe_eval_gpu_multiplier 2 \
        --vllm_enforce_eager \
        --deepspeed_zpg 1 \
        --hf_entity allenai \
        --wandb_entity ai2-llm