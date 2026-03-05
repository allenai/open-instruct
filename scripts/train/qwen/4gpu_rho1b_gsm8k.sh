#!/bin/bash

EXP_NAME="rho1b_gsm8k"
MODEL_NAME_OR_PATH="realtreetune/rho-1b-sft-GSM8K"
DATASETS="mnoukhov/rlvr_gsm8k_treetune_prompt 1.0"
BEAKER_IMAGE="michaeln/open_instruct"

LOCAL_EVALS="mnoukhov/rlvr_gsm8k_treetune_prompt 1.0"
LOCAL_EVAL_SPLITS="test"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/saturn \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus 4 \
    --budget ai2/oe-adapt \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run --active open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --run_name $EXP_NAME \
    --local_eval_every 100 \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --beta 0.1 \
    --async_steps 1 \
    --inflight_updates \
    --filter_zero_std_samples False \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name simple_concat_with_space \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --num_learners_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --with_tracking \
    --push_to_hub False $@
