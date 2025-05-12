# try rubric-based eval using this dataset: ai2-adapt-dev/tulu-3-sft-50k-criteria-gpt4o-classified
# mix with hamish rewritten math problems: ai2-adapt-dev/tulu-3-sft-57k-criteria-gpt4o-classified-rewritten-math
# mix with rewritten math, shortqa and tuu 3 ifeval rlvr: ai2-adapt-dev/tulu-3-sft-72k-criteria-gpt4o-classified-rewritten-math-shortqa-ifeval
# rubric+ref+math+shortqa_ifeval: ai2-adapt-dev/tulu-3-sft-92k-criteria-gpt4o-classified-rewritten-math-shortqa-ifeval-v2
# previous tested dataset: faezeb/tulu-3-sft-t3-70b-thinker-sampled
# tulu 3 rewritten replace string f1 with lm jduge + ifeval: ai2-adapt-dev/tulu_3_rewritten_100k-v2-ifeval

python open_instruct/grpo_fast_wip.py \
    --dataset_mixer_list ai2-adapt-dev/tulu_3_rewritten_100k-v2-ifeval 512 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/tulu_3_rewritten_100k-v2-ifeval 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --stop_strings "</answer>" \
    --apply_llm_verifier_reward true \
    --apply_verifiable_reward false \
    --apply_r1_style_format_reward true \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu_thinker_r1_style \
    --learning_rate 3e-7 \
    --total_episodes 5000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --num_evals 20 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.5 \
    --save_traces \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --llm_judge_model "gpt-4.1-standard" \
    --llm_judge_type "quality_rubric"
    # --with_tracking

    # --chat_template_name r1_simple_chat_postpend_think \ Qwen/Qwen2.5-0.5B
    #     --model_name_or_path HuggingFaceTB/SmolLM2-135M \

    # potential wildchat subset allenai/tulu-3-wildchat-reused-on-policy-8b
    #     --apply_llm_verifier_reward true \
    

        # --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 64 \
        # --dataset_mixer_eval_list ai2-adapt-dev/rlvr_gsm8k_zs 16 \
        # --apply_r1_style_format_reward \
        # --apply_verifiable_reward true \




# python open_instruct/grpo_fast_wip.py \
#     --dataset_mixer_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 5000 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 64 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 10240 \
#     --max_prompt_token_length 2048 \
#     --response_length 8192 \
#     --pack_length 16384 \
#     --per_device_train_batch_size 1 \
#     --num_unique_prompts_rollout 16 \
#     --num_samples_per_prompt_rollout 4 \
#     --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
#     --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
#     --output_dir /weka/oe-adapt-default/faezeb/model_checkpoints/${exp_name} \
#     --stop_strings "</answer>" \
#     --apply_llm_verifier_reward true \
#     --apply_verifiable_reward false \
#     --non_stop_penalty True \
#     --temperature 0.7 \
#     --ground_truths_key ground_truth \
#     --oe_eval_max_length 8192 \
#     --chat_template_name tulu_thinker_r1_style \
#     --learning_rate 5e-6 \
#     --total_episodes 1000 \
#     --deepspeed_stage 2 \
#     --per_device_train_batch_size 1 \
#     --num_mini_batches 1 \
#     --num_learners_per_node 8 8 \
#     --num_epochs 1 \
#     --beta 0.01 \
#     --vllm_tensor_parallel_size 1 \
#     --vllm_num_engines 16 \
#     --lr_scheduler_type constant \
#     --seed 1 \
#     --num_evals 100 \
#     --save_freq 100 \
#     --try_launch_beaker_eval_jobs_on_weka True \
#     --gradient_checkpointing \
#     --with_tracking \
#     --hf_entity allenai \
#     --wandb_entity ai2-llm