#!/bin/bash

# OLMo 2.5 model
MODEL_NAME_OR_PATH="/weka/oe-eval-default/ai2-llm/checkpoints/lucas/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_yarn-fullonly_50B-740666e3/step11921-hf"
GS_MODEL_NAME="olmo25_7b_lc_beta_740666e3"

# english only DAPO
DATASETS="mnoukhov/DAPO-Math-14k-Processed-RLVR 1.0"

# math evals
EVALS="minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek"

# AIME 2024, 2025 local evals
LOCAL_EVALS="mnoukhov/aime2024-25-rlvr 1.0 mnoukhov/aime2024-25-rlvr 1.0"
LOCAL_EVAL_SPLITS="test_2024 test_2024 test_2025 test_2025"

EXP_NAME="grpo_dapo14k_${gs_model_name}"

cluster=ai2/augusta

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${cluster} \
    --workspace ai2/tulu-thinker \
    --priority high \
    --pure_docker_mode \
    --image ${1:-michaeln/open_instruct_olmo2_retrofit} \
    --preemptible \
    --num_nodes 6 \
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
    --num_unique_prompts_rollout 128 \
    --num_mini_batches 4 \
    --num_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_token_length 18432 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 32768 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker_r1_style_nochat \
    --stop_strings "</answer>" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 2048000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 40 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 25 \
    --save_freq 25 \
    --checkpoint_state_freq 25 \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --oe_eval_max_length 32000 \
    --try_launch_beaker_eval_jobs_on_weka True \
    --oe_eval_tasks $EVALS \
    --oe_eval_beaker_image oe-eval-beaker/oe_eval_olmo2_retrofit_auto
