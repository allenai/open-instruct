#!/bin/bash

# saurab command:
exp_name="0410_qwen2.5_7B_thinker_grpo_fast_llm_judge_gpt4o-mini_quality_ref_${RANDOM}_gpt-4o"

python mason.py \
    --description $exp_name \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --image saurabhs/async_llm_judges --pure_docker_mode \
    --priority high \
    --preemptible \
    --num_nodes 4 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast_wip.py \
    --dataset_mixer_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 20000 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 64 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 128 \
    --num_samples_per_prompt_rollout 8 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,alpaca_eval_v2::tulu \
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
    --total_episodes 2000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 8 8 \
    --num_epochs 1 \
    --beta 0.01 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 16 \
    --lr_scheduler_type constant \
    --seed 1 \
    --num_evals 50 \
    --save_freq 50 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --gradient_checkpointing \
    --with_tracking \
    --hf_entity allenai \
    --wandb_entity ai2-llm

    # --beaker_datasets /models:01JQAWR48DRN9XCHX6YQCG9RF8 \


# # add date to the experiment name
# exp_name="0408_qwen2.5_7B_thinker_grpo_fast_llm_judge_gpt4o-mini_quality_ref_${RANDOM}_async_"
# # exp_name="test_deubg_not_docker"
# python mason.py \
#     --description $exp_name \
#     --cluster ai2/jupiter-cirrascale-2 \
#     --workspace ai2/tulu-3-dev \
#     --priority high \
#     --preemptible \
#     --num_nodes 4 \
#     --max_retries 0 \
#     --budget ai2/oe-adapt \
#     --beaker_datasets /models:01JQAWR48DRN9XCHX6YQCG9RF8 \
#     --image ai2/cuda11.8-cudnn8-dev-ubuntu20.04 \
#     --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast_wip.py \
#     --dataset_mixer_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 20000 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list faezeb/tulu-3-sft-t3-70b-thinker-sampled 64 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 10240 \
#     --max_prompt_token_length 2048 \
#     --response_length 8192 \
#     --pack_length 16384 \
#     --per_device_train_batch_size 1 \
#     --num_unique_prompts_rollout 128 \
#     --num_samples_per_prompt_rollout 8 \
#     --model_name_or_path /models \
#     --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,alpaca_eval_v2::tulu \
#     --output_dir /weka/oe-adapt-default/faezeb/model_checkpoints/${exp_name} \
#     --apply_llm_verifier_reward true \
#     --apply_verifiable_reward false \
#     --apply_r1_style_format_reward true \
#     --non_stop_penalty True \
#     --non_stop_penalty_value 0.0 \
#     --temperature 0.7 \
#     --ground_truths_key ground_truth \
#     --oe_eval_max_length 8192 \
#     --chat_template_name tulu_thinker_r1_style \
#     --kl_estimator kl3 \
#     --learning_rate 5e-6 \
#     --total_episodes 2000000 \
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
#     --num_evals 50 \
#     --save_freq 50 \
#     --try_launch_beaker_eval_jobs_on_weka True \
#     --gradient_checkpointing \
#     --with_tracking \
#     --hf_entity allenai \
#     --wandb_entity ai2-llm
#     # --stop_strings "</answer>" \
#     # ai2/ceres-cirrascale ai2/jupiter-cirrascale-2 
#     # --pure_docker_mode \
#     # --image faezeb/open_instruct_grpo_lm_judge_async

# older image: faezeb/open_instruct_llm_judge_0407
# --image ai2/cuda11.8-cudnn8-dev-ubuntu20.04

# exp_name="test_${RANDOM}"
# kl_estimator="kl3"
# beta=0.01
# learning_rate=5e-6
# num_unique_prompts_rollout=16
# nspp=4
# dataset="hamishivi/tulu-3-sft-t3-70b-thinker"
# python mason.py \
#     --cluster ai2/ceres-cirrascale ai2/jupiter-cirrascale-2 \
#     --workspace ai2/tulu-3-dev \
#     --priority high \
#     --preemptible \
#     --num_nodes 2 \
#     --max_retries 0 \
#     --budget ai2/oe-adapt \
#     --pure_docker_mode \
#     --image hamishivi/open_instruct_tulu_thinker5 \
#     --beaker_datasets /models:01JQAWR48DRN9XCHX6YQCG9RF8 \
#     --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/grpo_fast.py \
#     --exp_name $exp_name \
#     --beta $beta \
#     --num_samples_per_prompt_rollout $nspp \
#     --num_unique_prompts_rollout $num_unique_prompts_rollout \
#     --output_dir /weka/oe-adapt-default/faezeb/model_checkpoints/${exp_name} \
#     --kl_estimator $kl_estimator \
#     --learning_rate $learning_rate \
#     --dataset_mixer_list $dataset 256 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list $dataset 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 10240 \
#     --max_prompt_token_length 2048 \
#     --response_length 8192 \
#     --pack_length 16384 \
#     --model_name_or_path /models \
#     --apply_verifiable_reward True \
#     --non_stop_penalty True \
#     --non_stop_penalty_value 0.0 \
#     --chat_template_name tulu \
#     --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
#     --oe_eval_max_length 8192 \
#     --temperature 1.0 \
#     --total_episodes 200 \
#     --deepspeed_stage 2 \
#     --per_device_train_batch_size 1 \
#     --num_mini_batches 1 \
#     --num_learners_per_node 8 8 \
#     --num_epochs 1 \
#     --vllm_tensor_parallel_size 1 \
#     --vllm_num_engines 24 \
#     --lr_scheduler_type constant \
#     --seed 1 \
#     --num_evals 100 \
#     --save_freq 100 \
#     --try_launch_beaker_eval_jobs_on_weka True \
#     --gradient_checkpointing \
#     --with_tracking

    # --pure_docker_mode \
    # --image nathanl/open_instruct_auto \