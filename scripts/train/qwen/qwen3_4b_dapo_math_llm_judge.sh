#!/bin/bash

set -euo pipefail

EXP_NAME="${EXP_NAME:-qwen3_4b_base_dapo_llm_judge}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Base}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-hosted_vllm/Qwen/Qwen3-4B-Instruct}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"

DATASETS="${DATASETS:-mnoukhov/dapo_math_14k_en_openinstruct 1.0}"
DATASET_SPLITS="${DATASET_SPLITS:-train}"

LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/aime_2025_openinstruct 1.0 mnoukhov/brumo_2025_openinstruct 1.0}"
LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-train}"

CLUSTER="${CLUSTER:-ai2/saturn ai2/jupiter ai2/neptune}"
PRIORITY="${PRIORITY:-high}"

JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8001}"
JUDGE_SERVER_MAX_MODEL_LEN="${JUDGE_SERVER_MAX_MODEL_LEN:-32768}"
JUDGE_SERVER_GPU_MEMORY_UTILIZATION="${JUDGE_SERVER_GPU_MEMORY_UTILIZATION:-0.5}"
JUDGE_SERVER_TENSOR_PARALLEL_SIZE="${JUDGE_SERVER_TENSOR_PARALLEL_SIZE:-1}"
JUDGE_SERVER_LOG="${JUDGE_SERVER_LOG:-/tmp/qwen3_4b_judge_vllm.log}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-5}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

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
    --gpus 8 \
    --budget ai2/oe-adapt \
    -- bash -lc "
set -euo pipefail
export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export UV_LINK_MODE=\${UV_LINK_MODE:-copy}
export HOSTED_VLLM_API_BASE=http://127.0.0.1:${JUDGE_SERVER_PORT}/v1
JUDGE_SERVER_PID=

cleanup() {
    if [[ -n \"\${JUDGE_SERVER_PID}\" ]]; then
        kill \"\${JUDGE_SERVER_PID}\" >/dev/null 2>&1 || true
        wait \"\${JUDGE_SERVER_PID}\" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

if [[ -n \"\${JUDGE_SERVER_VISIBLE_DEVICES:-}\" ]]; then
    JUDGE_VISIBLE_DEVICES=\"\${JUDGE_SERVER_VISIBLE_DEVICES}\"
elif [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ]]; then
    IFS=',' read -r -a _visible_devices <<< \"\${CUDA_VISIBLE_DEVICES}\"
    _device_count=\${#_visible_devices[@]}
    _judge_device_count=${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}
    if (( _device_count < _judge_device_count + 2 )); then
        echo \"Need at least \$(( _judge_device_count + 2 )) visible GPUs to leave room for learners and judge, found \${_device_count}\" >&2
        exit 1
    fi
    _start_idx=\$(( _device_count - _judge_device_count ))
    JUDGE_VISIBLE_DEVICES=\"\$(IFS=,; echo \"\${_visible_devices[*]:_start_idx:_judge_device_count}\")\"
else
    mapfile -t _all_devices < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    _device_count=\${#_all_devices[@]}
    _judge_device_count=${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}
    if (( _device_count < _judge_device_count + 2 )); then
        echo \"Need at least \$(( _judge_device_count + 2 )) GPUs to leave room for learners and judge, found \${_device_count}\" >&2
        exit 1
    fi
    _start_idx=\$(( _device_count - _judge_device_count ))
    JUDGE_VISIBLE_DEVICES=\"\$(IFS=,; echo \"\${_all_devices[*]:_start_idx:_judge_device_count}\")\"
fi

echo \"Using judge CUDA_VISIBLE_DEVICES=\${JUDGE_VISIBLE_DEVICES}\" >&2

CUDA_VISIBLE_DEVICES=\${JUDGE_VISIBLE_DEVICES} uv run python -m vllm.entrypoints.openai.api_server \
    --model \"${LLM_JUDGE_MODEL#hosted_vllm/}\" \
    --port \"${JUDGE_SERVER_PORT}\" \
    --tensor-parallel-size \"${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}\" \
    --max-model-len \"${JUDGE_SERVER_MAX_MODEL_LEN}\" \
    --gpu-memory-utilization \"${JUDGE_SERVER_GPU_MEMORY_UTILIZATION}\" \
    >\"${JUDGE_SERVER_LOG}\" 2>&1 &
JUDGE_SERVER_PID=\$!

for _ in \$(seq 1 60); do
    if curl -fsS \"\${HOSTED_VLLM_API_BASE}/models\" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! curl -fsS \"\${HOSTED_VLLM_API_BASE}/models\" >/dev/null 2>&1; then
    echo \"Judge vLLM server failed to start; see ${JUDGE_SERVER_LOG}\" >&2
    exit 1
fi

UV_LINK_MODE=copy uv run --active open_instruct/grpo_fast.py \
    --run_name \"${RUN_NAME}\" \
    --exp_name \"${EXP_NAME}\" \
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
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 128000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 2 \
    --vllm_num_engines ${VLLM_NUM_ENGINES} \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --llm_judge_model \"${LLM_JUDGE_MODEL}\" \
    --llm_judge_override_verifier math \
    --llm_judge_max_tokens 256 \
    --llm_judge_temperature 0.0 \
    --llm_judge_max_context_length \"${JUDGE_SERVER_MAX_MODEL_LEN}\" \
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
    --push_to_hub False \
    \"\$@\"
" -- "$@"
