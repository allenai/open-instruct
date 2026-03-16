#!/bin/bash

set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_llm_judge}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-hosted_vllm/opencompass/CompassVerifier-3B}"
LLM_JUDGE_FALLBACK_VERIFIER="${LLM_JUDGE_FALLBACK_VERIFIER:-math}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"

DATASETS="${DATASETS:-mnoukhov/dapo_math_14k_en_openinstruct 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/aime_2025_openinstruct 1.0 mnoukhov/brumo_2025_openinstruct 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-train}"

CLUSTER="${CLUSTER:-ai2/saturn ai2/jupiter ai2/neptune}"
PRIORITY="${PRIORITY:-high}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-6}"
JUDGE_SERVER_MAX_MODEL_LEN="${JUDGE_SERVER_MAX_MODEL_LEN:-32768}"
JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8001}"
JUDGE_WORKSPACE="${JUDGE_WORKSPACE:-ai2/oe-adapt-code}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/judge_configs/compass_verifier_3b_judge.yaml}"
JUDGE_EXPERIMENT_NAME="${JUDGE_EXPERIMENT_NAME:-${RUN_NAME}_judge}"
JUDGE_EXPERIMENT_ID="${JUDGE_EXPERIMENT_ID:-}"

extract_json_field() {
    local expr="$1"
    python -c 'import json,sys; data=json.load(sys.stdin); print(eval(sys.argv[1], {"__builtins__": {}}, {"data": data}))' "${expr}"
}

if [[ -z "${JUDGE_EXPERIMENT_ID}" ]]; then
    judge_create_json="$(beaker experiment create "${JUDGE_CONFIG}" --name "${JUDGE_EXPERIMENT_NAME}" --workspace "${JUDGE_WORKSPACE}" --format json)"
    JUDGE_EXPERIMENT_ID="$(printf '%s\n' "${judge_create_json}" | extract_json_field 'data[0]["id"]')"
fi

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND=FLASHINFER \
    --env JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT}" \
    --env JUDGE_WORKSPACE="${JUDGE_WORKSPACE}" \
    --env JUDGE_CONFIG="${JUDGE_CONFIG}" \
    --env JUDGE_EXPERIMENT_NAME="${JUDGE_EXPERIMENT_NAME}" \
    --env JUDGE_EXPERIMENT_ID="${JUDGE_EXPERIMENT_ID}" \
    --env HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-}" \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- source scripts/train/qwen/setup_compass_verifier_judge.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --eval_pass_at_k 32 \
    --eval_top_p 0.95 \
    --vllm_top_p 1.0 \
    --local_eval_every 100 \
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
    --dataset_mixer_list ${DATASETS} \
    --dataset_mixer_list_splits ${DATASET_SPLITS} \
    --dataset_mixer_eval_list ${LOCAL_EVALS} \
    --dataset_mixer_eval_list_splits ${LOCAL_EVAL_SPLITS} \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 2 \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --llm_judge_fallback_verifier "${LLM_JUDGE_FALLBACK_VERIFIER}" \
    --llm_judge_model "${LLM_JUDGE_MODEL}" \
    --llm_judge_max_tokens 32 \
    --llm_judge_temperature 0.0 \
    --llm_judge_max_context_length "${JUDGE_SERVER_MAX_MODEL_LEN}" \
    --llm_judge_timeout 240 \
    --seed 1 \
    --save_freq 100 \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --with_tracking \
    --send_slack_alerts \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --chat_template qwen_instruct_user_boxed_math \
    --load_ref_policy True \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False "$@"
