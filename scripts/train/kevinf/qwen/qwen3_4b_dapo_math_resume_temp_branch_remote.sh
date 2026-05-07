#!/bin/bash
set -euo pipefail

if [[ -z "${BASE_CHECKPOINT_STATE_DIR:-}" ]]; then
    echo "BASE_CHECKPOINT_STATE_DIR must point to the source checkpoint-state directory."
    exit 1
fi

if [[ -z "${BRANCH_CHECKPOINT_STATE_DIR:-}" ]]; then
    echo "BRANCH_CHECKPOINT_STATE_DIR must point to the destination checkpoint-state directory."
    exit 1
fi

if [[ -z "${BRANCH_OUTPUT_DIR:-}" ]]; then
    echo "BRANCH_OUTPUT_DIR must point to a durable Weka output directory."
    exit 1
fi

if [[ "${BRANCH_OUTPUT_DIR}" != /weka/* && "${BRANCH_OUTPUT_DIR}" != /output* ]]; then
    echo "BRANCH_OUTPUT_DIR must be durable (/weka/... or /output...), got ${BRANCH_OUTPUT_DIR}"
    exit 1
fi

if [[ -z "${TEMPERATURE:-}" ]]; then
    echo "TEMPERATURE must be set for this continuation branch."
    exit 1
fi

BASE_STEP_TAG="${BASE_STEP_TAG:-global_step500}"
EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_temp_fork}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_t${TEMPERATURE}}"

if [[ ! -d "${BASE_CHECKPOINT_STATE_DIR}/${BASE_STEP_TAG}" ]]; then
    echo "Missing source checkpoint ${BASE_CHECKPOINT_STATE_DIR}/${BASE_STEP_TAG}"
    exit 1
fi

if [[ ! -d "${BRANCH_CHECKPOINT_STATE_DIR}/${BASE_STEP_TAG}" ]]; then
    mkdir -p "${BRANCH_CHECKPOINT_STATE_DIR}"
    cp -a "${BASE_CHECKPOINT_STATE_DIR}/${BASE_STEP_TAG}" "${BRANCH_CHECKPOINT_STATE_DIR}/${BASE_STEP_TAG}"
    echo "${BASE_STEP_TAG}" > "${BRANCH_CHECKPOINT_STATE_DIR}/latest"
fi

uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_pass_at_k 32 \
    --eval_top_p 0.95 \
    --vllm_top_p 1.0 \
    --beta 0.0 \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list hamishivi/DAPO-Math-17k-Processed_filtered 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/aime_2025_openinstruct 1.0 allenai/brumo_2025_openinstruct 1.0 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --non_stop_penalty False \
    --temperature "${TEMPERATURE}" \
    --temperature_schedule constant \
    --total_episodes 128000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --checkpoint_state_dir "${BRANCH_CHECKPOINT_STATE_DIR}" \
    --output_dir "${BRANCH_OUTPUT_DIR}" \
    --gradient_checkpointing \
    --with_tracking \
    --send_slack_alerts \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --chat_template qwen_instruct_user_boxed_math \
    --load_ref_policy False \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False \
    "$@"
