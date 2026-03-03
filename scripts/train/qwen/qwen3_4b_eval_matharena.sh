#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen3_4b_think_eval_matharena}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Thinking-2507}"
BEAKER_IMAGE="michaeln/open_instruct"

DATASETS="${DATASETS:-mnoukhov/brumo_2025_openinstruct 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/brumo_2025_openinstruct 1.0 mnoukhov/hmmt_feb_2025_openinstruct 1.0 mnoukhov/hmmt_nov_2025_openinstruct 1.0 mnoukhov/aime_2025_openinstruct 1.0}"
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
    --eval_on_step_0 \
    --eval_only \
    --eval_pass_at_k 16 \
    --eval_temperature 1.0 \
    --eval_top_p 0.95 \
    --eval_response_length 65536 \
    --vllm_top_p 1.0 \
    --vllm_num_engines 4 \
    --local_eval_every 100 \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --beta 0.0 \
    --async_steps 1 \
    --filter_zero_std_samples False \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 16 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --max_prompt_token_length 2048 \
    --response_length 65536 \
    --pack_length 70000 \
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
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --with_tracking \
    --total_episodes 128 \
    --push_to_hub False $@

    # --checkpoint_state_freq 200 \
    # --keep_last_n_checkpoints -1 \
