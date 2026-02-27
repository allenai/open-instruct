#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen25_05b_it_gsm8k_quartiles}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
BEAKER_IMAGE="michaeln/open_instruct"

DATASETS="${DATASETS:-mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-0 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-25 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-50 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-75 8}"
LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-0 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-25 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-50 8 mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct-75 8}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-train}"

CLUSTER="${CLUSTER:-ai2/saturn ai2/jupiter ai2/neptune}"
PRIORITY="${PRIORITY:-high}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASHINFER" \
    --gpus 4 \
    --budget ai2/oe-adapt \
    -- \
uv run --active open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_pass_at_k 128 \
    --vllm_top_p 1.0 \
    --local_eval_every 100 \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --filter_zero_std_samples False \
    --log_train_solve_rate_metrics True \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 4096 \
    --pack_length 8192 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name qwen_instruct_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 256000 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --save_freq 200 \
    --vllm_enable_prefix_caching \
    --num_learners_per_node 1 \
    --vllm_num_engines 3 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy True \
    --with_tracking \
    --push_to_hub False $@

    # --checkpoint_state_freq 200 \
    # --keep_last_n_checkpoints -1 \
