#!/bin/bash

set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_llm_judge}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-hosted_vllm/opencompass/CompassVerifier-3B}"
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
JUDGE_EXPERIMENT_ID="${JUDGE_EXPERIMENT_ID:-}"

extract_json_field() {
    local expr="$1"
    python -c 'import json,sys; data=json.load(sys.stdin); print(eval(sys.argv[1], {"__builtins__": {}}, {"data": data}))' "${expr}"
}

if [[ -z "${HOSTED_VLLM_API_BASE:-}" ]]; then
    if [[ -z "${JUDGE_EXPERIMENT_ID}" ]]; then
        echo "Set HOSTED_VLLM_API_BASE or JUDGE_EXPERIMENT_ID for an already-running judge." >&2
        exit 1
    fi

    judge_tasks_json="$(beaker experiment tasks "${JUDGE_EXPERIMENT_ID}" --format json)"
    JUDGE_TASK_NAME="$(printf '%s\n' "${judge_tasks_json}" | extract_json_field 'data[0]["name"]')"
    JUDGE_JOB_ID="$(printf '%s\n' "${judge_tasks_json}" | extract_json_field 'data[0]["jobs"][0]["id"]')"

    beaker experiment await "${JUDGE_EXPERIMENT_ID}" "${JUDGE_TASK_NAME}" started --timeout 10m >/dev/null

    judge_job_json="$(beaker job get "${JUDGE_JOB_ID}" --format json)"
    JUDGE_HOSTNAME="$(printf '%s\n' "${judge_job_json}" | extract_json_field 'next(env["value"] for env in data[0]["execution"]["spec"]["envVars"] if env["name"] == "BEAKER_NODE_HOSTNAME")')"
    export HOSTED_VLLM_API_BASE="http://${JUDGE_HOSTNAME}:${JUDGE_SERVER_PORT}/v1"

    for _ in $(seq 1 90); do
        if curl -fsS "${HOSTED_VLLM_API_BASE}/models" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    if ! curl -fsS "${HOSTED_VLLM_API_BASE}/models" >/dev/null 2>&1; then
        echo "Existing judge server failed to become healthy at ${HOSTED_VLLM_API_BASE}" >&2
        exit 1
    fi
fi

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --cluster ${CLUSTER} \
    --workspace ai2/oe-adapt-code \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND=FLASHINFER \
    --env HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE}" \
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- \
uv run open_instruct/grpo_fast.py \
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
    --remap_verifier math=general-compass_verifier \
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
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions False \
    --chat_template qwen_instruct_user_boxed_math \
    --load_ref_policy True \
    --keep_last_n_checkpoints -1 \
    --push_to_hub False "$@"
