#!/bin/bash
# older data:  faezeb/tulu-3-sft-t3-70b-thinker-sampled
# rubric+ref+math+shortqa_ifeval: ai2-adapt-dev/tulu-3-sft-92k-criteria-gpt4o-classified-rewritten-math-shortqa-ifeval-v2
# ref-based / ref+math+shortqa+ifeval: ai2-adapt-dev/tulu-3-sft-92k-classified-rewritten-math-shortqa-ifeval-ref-based
# tulu3-rewritten with ref base lm judge + ifeval: ai2-adapt-dev/tulu_3_rewritten_100k-v2-ifeval
# general-thoughts-100k-rewritten with ref base lm judge + ifeval: ai2-adapt-dev/general-thoughts-100k-rewritten-v2-ifeval

# saurab command:
# exp_name="0504_qwen2.5_7B_thinker_grpo_fast_llm_judge__gpt-4o_quality_ref_${RANDOM}_large"
exp_name="0508_qwen2.5_7B_thinker_grpo_lm_judge_gpt-41_ref_${RANDOM}_generalthought_rewritten_ifeval"
# exp_name="0505_qwen2.5_7B_thinker_grpo_fast_llm_judge_gpt-4o_quality_ref_3188_v2/resume_from_step_250/"
model_weka_path="/weka/oe-adapt-default/faezeb/model_checkpoints/0505_qwen2.5_7B_thinker_grpo_fast_llm_judge_gpt-4o_quality_ref_3188_v2/grpo_fast_wip__1__1746517521_checkpoints/step_250"
python mason.py \
    --description $exp_name \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-results \
    --priority high \
    --preemptible \
    --num_nodes 4 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --image ai2/cuda11.8-cudnn8-dev-ubuntu20.04 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast_wip.py \
    --dataset_mixer_list ai2-adapt-dev/general-thoughts-100k-rewritten-v2-ifeval 188000 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/general-thoughts-100k-rewritten-v2-ifeval 128 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 128 \
    --num_samples_per_prompt_rollout 8 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,alpaca_eval_v2::tulu,ifeval::tulu,popqa::tulu,drop::llama3,codex_humanevalplus::tulu,mmlu:cot::summarize \
    --output_dir /weka/oe-adapt-default/faezeb/model_checkpoints/${exp_name} \
    --apply_llm_verifier_reward true \
    --apply_verifiable_reward false \
    --apply_r1_style_format_reward true \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --oe_eval_max_length 8192 \
    --chat_template_name tulu_thinker_r1_style \
    --kl_estimator kl3 \
    --learning_rate 5e-6 \
    --total_episodes 520000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_learners_per_node 6 \
    --num_epochs 1 \
    --beta 0.01 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 10 \
    --lr_scheduler_type constant \
    --seed 1 \
    --num_evals 50 \
    --save_freq 50 \
    --save_traces \
    --try_launch_beaker_eval_jobs_on_weka True \
    --gradient_checkpointing \
    --with_tracking \
    --hf_entity allenai \
    --wandb_entity ai2-llm \
    --llm_judge_model "gpt-4.1-standard"