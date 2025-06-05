#!/bin/bash
# older data:  faezeb/tulu-3-sft-t3-70b-thinker-sampled
# rubric+ref+math+shortqa_ifeval: ai2-adapt-dev/tulu-3-sft-92k-criteria-gpt4o-classified-rewritten-math-shortqa-ifeval-v2
# ref-based / ref+math+shortqa+ifeval: ai2-adapt-dev/tulu-3-sft-92k-classified-rewritten-math-shortqa-ifeval-ref-based
# tulu3-rewritten with ref base lm judge + ifeval: ai2-adapt-dev/tulu_3_rewritten_100k-v2-ifeval
# general-thoughts-100k-rewritten with ref base lm judge + ifeval: ai2-adapt-dev/general-thoughts-100k-rewritten-v2-ifeval

reward_model_revision="rm_olmo_2_7b_lc_base_2e-5_1_skyworkstulufull__1__1749015683"
# reward_model_revision="main"
# exp_name="0504_qwen2.5_7B_thinker_grpo_fast_llm_judge__gpt-4o_quality_ref_${RANDOM}_large"
# exp_name="0508_qwen2.5_7B_thinker_grpo_rm_judge_${RANDOM}_tulu_3_rewritten_no_verifiable"
exp_name="grpo_with_rm_olmo_2_lc_non_verifiable"
python mason.py \
    --description $exp_name \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-thinker \
    --priority high \
    --preemptible \
    --num_nodes 4 \
    --max_retries 0 \
    --no_auto_dataset_cache \
    --budget ai2/oe-adapt \
    --image ai2/cuda11.8-cudnn8-dev-ubuntu20.04 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name test_grpo_rm_with_valpy_code \
    --beta 0.01 \
    --local_mini_batch_size 16 \
    --number_samples_per_prompt 8 \
    --local_rollout_batch_size 2 \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list jacobmorrison/tulu_3_rewritten_53k_no_verifiable 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list jacobmorrison/tulu_3_rewritten_53k_no_verifiable 128 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,alpaca_eval_v2::tulu,ifeval::tulu,popqa::tulu,drop::llama3,codex_humanevalplus::tulu,mmlu:cot::summarize \
    --non_stop_penalty \
    --stop_strings '</answer>' \
    --stop_token eos \
    --penalty_reward_value 0.0 \
    --temperature 0.7 \
    --chat_template_name tulu_thinker_r1_style \
    --total_episodes 2000000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --num_mini_batches 2 \
    --actor_num_gpus_per_node 4 8 8 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --add_r1_style_format_reward true \
    --vllm_num_engines 4 \
    --vllm_gpu_memory_utilization 0.9 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward false \
    --seed 1 \
    --num_evals 50 \
    --save_freq 50 \
    --reward_model_multiplier 1.0 \
    --apply_reward_model true \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision $reward_model_revision \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking \
    --hf_entity allenai \
    --wandb_entity ai2-llm \
    --add_bos
    
    # ai2/cuda11.8-cudnn8-dev-ubuntu20.04
    # nathanl/open_instruct_auto
    
    # python open_instruct/grpo_fast.py \
    # --dataset_mixer_list jacobmorrison/tulu_3_rewritten_53k_no_verifiable 5000 \
    # --dataset_mixer_list_splits train \
    # --dataset_mixer_eval_list jacobmorrison/tulu_3_rewritten_53k_no_verifiable 128 \
    # --dataset_mixer_eval_list_splits train \
    # --max_token_length 10240 \
    # --max_prompt_token_length 2048 \
    # --response_length 8192 \
    # --pack_length 16384 \
    # --per_device_train_batch_size 1 \
    # --num_unique_prompts_rollout 16 \
    # --num_samples_per_prompt_rollout 8 \
    # --model_name_or_path Qwen/Qwen2.5-0.5B \
    # --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,alpaca_eval_v2::tulu,ifeval::tulu,popqa::tulu,drop::llama3,codex_humanevalplus::tulu,mmlu:cot::summarize \
    # --output_dir /weka/oe-adapt-default/jacobm/llm-as-a-judge/checkpoints/rm-vs-llm/${exp_name} \
    # --apply_verifiable_reward false \
    # --apply_r1_style_format_reward true \
    # --reward_model_path allenai/open_instruct_dev \
    # --reward_model_revision $reward_model_revision \
    # --reward_model_tokenizer_path allenai/open_instruct_dev \
    # --reward_model_tokenizer_revision $reward_model_revision \
    # --non_stop_penalty True \
    # --non_stop_penalty_value 0.0 \
    # --temperature 0.7 \
    # --ground_truths_key ground_truth \
    # --oe_eval_max_length 8192 \
    # --chat_template_name tulu_thinker_r1_style \
    # --use_reward_model true \
    # --kl_estimator kl3 \
    # --learning_rate 5e-6 \
    # --total_episodes 520000 \
    # --deepspeed_stage 2 \
    # --per_device_train_batch_size 1 \
    # --num_mini_batches 2 \
    # --num_learners_per_node 6 \
    # --num_epochs 1 \
    # --beta 0.01 \
    # --vllm_tensor_parallel_size 1 \
    # --vllm_num_engines 2 \
    # --lr_scheduler_type constant \
    # --seed 1 \
    # --num_evals 50 \
    # --save_freq 50 \
    # --save_traces \
    # --try_launch_beaker_eval_jobs_on_weka True \
    # --gradient_checkpointing \
    # --with_tracking \
    # --hf_entity allenai \
    # --wandb_entity ai2-llm 