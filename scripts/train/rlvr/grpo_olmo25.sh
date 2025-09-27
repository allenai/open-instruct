#!/bin/bash

# OLMo 2.5 model
MODEL_NAME_OR_PATH="/weka/oe-training-default/ai2-llm/checkpoints/tylerr/long-context/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_packing_yarn-fullonly_50B-fb13a737/step11921-hf"
GS_MODEL_NAME="olmo25_7b_lc_final_fb13a737"

# english only DAPO
DATASETS="mnoukhov/DAPO-Math-14k-Processed-RLVR 1.0 TTTXXX01/MATH_3000_Filtered 1.0"
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
EXP_NAME="grpo_dapoteng17k_3e6_${GS_MODEL_NAME}"

cluster=ai2/augusta

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --pure_docker_mode \
    --image michaeln/open_instruct_olmo2_retrofit \
    --preemptible \
    --num_nodes 5 \
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
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --inflight_updates \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 32000 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_dapo \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 32 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 25 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --oe_eval_max_length 32000 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --eval_priority high \
    --oe_eval_tasks $EVALS \
    --oe_eval_gpu_multiplier 4 \
    --oe_eval_beaker_image michaeln/oe_eval_olmo2_retrofit
