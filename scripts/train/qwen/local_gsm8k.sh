#!/bin/bash

EXP_NAME="${EXP_NAME:-qwen25_05b_it_gsm8k_quartiles}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
GSM8K_LLM_JUDGE_MODEL="${GSM8K_LLM_JUDGE_MODEL:-hosted_vllm/opencompass/CompassVerifier-3B}"
JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8001}"
JUDGE_SERVER_MAX_MODEL_LEN="${JUDGE_SERVER_MAX_MODEL_LEN:-2048}"
JUDGE_SERVER_GPU_MEMORY_UTILIZATION="${JUDGE_SERVER_GPU_MEMORY_UTILIZATION:-0.18}"
JUDGE_SERVER_LOG="${JUDGE_SERVER_LOG:-/tmp/compass_verifier_vllm.log}"
JUDGE_SERVER_PID=""

DATASETS="${DATASETS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-test}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test}"

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cleanup() {
    if [[ -n "${JUDGE_SERVER_PID}" ]]; then
        kill "${JUDGE_SERVER_PID}" >/dev/null 2>&1 || true
        wait "${JUDGE_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

if [[ "${GSM8K_LLM_JUDGE_MODEL}" == hosted_vllm/* && -z "${HOSTED_VLLM_API_BASE:-}" ]]; then
    export HOSTED_VLLM_API_BASE="http://127.0.0.1:${JUDGE_SERVER_PORT}/v1"
    JUDGE_MODEL_PATH="${GSM8K_LLM_JUDGE_MODEL#hosted_vllm/}"

    uv run python -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL_PATH}" \
        --port "${JUDGE_SERVER_PORT}" \
        --tensor-parallel-size 1 \
        --max-model-len "${JUDGE_SERVER_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${JUDGE_SERVER_GPU_MEMORY_UTILIZATION}" \
        >"${JUDGE_SERVER_LOG}" 2>&1 &
    JUDGE_SERVER_PID=$!

    for _ in $(seq 1 60); do
        if curl -fsS "${HOSTED_VLLM_API_BASE}/models" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    if ! curl -fsS "${HOSTED_VLLM_API_BASE}/models" >/dev/null 2>&1; then
        echo "CompassVerifier vLLM server failed to start; see ${JUDGE_SERVER_LOG}" >&2
        exit 1
    fi
fi

uv run --active open_instruct/grpo_fast.py \
    --output_dir results \
    --run_name ${RUN_NAME} \
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 1 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 512 \
    --response_length 4096 \
    --pack_length 8192 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name qwen_instruct_boxed_math \
    --non_stop_penalty False \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --total_episodes 128 \
    --deepspeed_stage 2 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --llm_judge_model "${GSM8K_LLM_JUDGE_MODEL}" \
    --llm_judge_override_verifier gsm8k \
    --llm_judge_max_tokens 32 \
    --llm_judge_temperature 0.0 \
    --llm_judge_max_context_length "${JUDGE_SERVER_MAX_MODEL_LEN}" \
    --llm_judge_timeout 120 \
    --seed 1 \
    --local_eval_every 1 \
    --save_freq 200 \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --single_gpu_mode \
    --vllm_enforce_eager \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --clip_higher 0.28 \
    --mask_truncated_completions False \
    --load_ref_policy False \
    --eval_on_step_0 True \
    --eval_pass_at_k 4 \
    --with_tracking False \
    --push_to_hub False $@

    # --checkpoint_state_freq 200 \
    # --keep_last_n_checkpoints -1 \

    # --eval_priority normal \
    # --try_launch_beaker_eval_jobs_on_weka True \
    # --oe_eval_max_length 32768 \
    # --oe_eval_gpu_multiplier 2  \
    # --oe_eval_beaker_image michaeln/oe_eval_internal \
    # --oe_eval_tasks $EVALS \
