#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen2.5_0.5b_instruct_gsm8k}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

BEAKER_IMAGE="nathanl/open_instruct"

CLUSTER="${CLUSTER:-ai2/neptune ai2/jupiter ai2/ceres}"
PRIORITY="${PRIORITY:-high}"
NUM_GPUS="${NUM_GPUS:-3}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    --gpus ${NUM_GPUS} \
    --budget ai2/oe-adapt \
    -- \
uv run --active open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --run_name ${RUN_NAME} \
    --beta 0.0 \
    --async_steps 1 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list mnoukhov/gsm8k-platinum-openinstruct 1.0 \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length 512 \
    --response_length 2048 \
    --pack_length 8192 \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --system_prompt_override_file scripts/train/qwen/math_system_prompt.txt \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --num_learners_per_node 2 \
    --vllm_num_engines 1 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --with_tracking \
    --send_slack_alerts \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False $@
