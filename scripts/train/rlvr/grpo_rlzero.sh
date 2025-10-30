#!/bin/bash

# OLMo 3 model
MODEL_NAME_OR_PATH="/weka/oe-adapt-default/michaeln/checkpoints/olmo3-7b-base"
GS_MODEL_NAME="olmo3_7b_base"

# english only DAPO
# DATASETS="mnoukhov/DAPO-Math-14k-Processed-RLVR 1.0 TTTXXX01/MATH_3000_Filtered 1.0"
DATASETS="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 1.0 saurabh5/MATH_3000_Filtered_olmo_completions_new_template_filtered 1.0"
# DATASETS="mnoukhov/deepscaler_20k_medhard_nolatex_rlvr 1.0"
# DATASETS=""

# math evals
# EVALS="minerva_math_500::hamish_zs_reasoning_deepseek"
EVALS="aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo"

# AIME 2024, 2025 local evals
LOCAL_EVALS="mnoukhov/aime2024-25-rlvr 1.0 mnoukhov/aime2024-25-rlvr 1.0"
LOCAL_EVAL_SPLITS="test_2024 test_2024 test_2025 test_2025"
# tengmath3k
# EXP_NAME="grpo_deepscaler20k_k8_${GS_MODEL_NAME}"
EXP_NAME="grpo_17kfilter_${GS_MODEL_NAME}"

cluster=ai2/jupiter

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image michaeln/open_instruct_rlzero \
    --preemptible \
    --num_nodes 9 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gs_model_name $GS_MODEL_NAME \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- \
source configs/beaker_configs/ray_node_setup.sh \&\& \
source configs/beaker_configs/code_api_setup.sh \&\& \
python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --active_fill_completions \
    --no_resample_solve_rate 0.9 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 12000 \
    --pack_length 32768 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_dapo \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 64 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --oe_eval_max_length 32768 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --eval_priority high \
    --oe_eval_tasks $EVALS \
    --oe_eval_gpu_multiplier 4 \
    --oe_eval_beaker_image michaeln/oe_eval_rlzero
