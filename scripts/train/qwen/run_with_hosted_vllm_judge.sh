#!/bin/bash

set -euo pipefail

LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-hosted_vllm/opencompass/CompassVerifier-3B}"
JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8001}"
JUDGE_SERVER_MAX_MODEL_LEN="${JUDGE_SERVER_MAX_MODEL_LEN:-8192}"
JUDGE_SERVER_GPU_MEMORY_UTILIZATION="${JUDGE_SERVER_GPU_MEMORY_UTILIZATION:-0.18}"
JUDGE_SERVER_TENSOR_PARALLEL_SIZE="${JUDGE_SERVER_TENSOR_PARALLEL_SIZE:-1}"
JUDGE_SERVER_LOG="${JUDGE_SERVER_LOG:-/tmp/compass_verifier_vllm.log}"
JUDGE_SERVER_PID=""

if [[ "${LLM_JUDGE_MODEL}" != hosted_vllm/* ]]; then
    echo "LLM_JUDGE_MODEL must use the hosted_vllm/ prefix to launch a local judge server." >&2
    exit 1
fi

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-http://127.0.0.1:${JUDGE_SERVER_PORT}/v1}"

cleanup() {
    if [[ -n "${JUDGE_SERVER_PID}" ]]; then
        kill "${JUDGE_SERVER_PID}" >/dev/null 2>&1 || true
        wait "${JUDGE_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

if [[ -n "${JUDGE_SERVER_VISIBLE_DEVICES:-}" ]]; then
    JUDGE_VISIBLE_DEVICES="${JUDGE_SERVER_VISIBLE_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    device_count="${#visible_devices[@]}"
    judge_device_count="${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}"
    if (( device_count < judge_device_count )); then
        echo "Need at least ${judge_device_count} visible GPUs for the judge, found ${device_count}" >&2
        exit 1
    fi
    start_idx=$(( device_count - judge_device_count ))
    JUDGE_VISIBLE_DEVICES="$(IFS=,; echo "${visible_devices[*]:start_idx:judge_device_count}")"
else
    mapfile -t all_devices < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    device_count="${#all_devices[@]}"
    judge_device_count="${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}"
    if (( device_count < judge_device_count )); then
        echo "Need at least ${judge_device_count} GPUs for the judge, found ${device_count}" >&2
        exit 1
    fi
    start_idx=$(( device_count - judge_device_count ))
    JUDGE_VISIBLE_DEVICES="$(IFS=,; echo "${all_devices[*]:start_idx:judge_device_count}")"
fi

echo "Using judge CUDA_VISIBLE_DEVICES=${JUDGE_VISIBLE_DEVICES}" >&2

CUDA_VISIBLE_DEVICES="${JUDGE_VISIBLE_DEVICES}" uv run python -m vllm.entrypoints.openai.api_server \
    --model "${LLM_JUDGE_MODEL#hosted_vllm/}" \
    --port "${JUDGE_SERVER_PORT}" \
    --tensor-parallel-size "${JUDGE_SERVER_TENSOR_PARALLEL_SIZE}" \
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

"$@"
