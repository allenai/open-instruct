#!/bin/bash

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export UV_CACHE_DIR="/tmp/uvcache"
export BALLSIM_API_URL="${BALLSIM_API_URL:-http://localhost:2345}"
export BALLSIM_API_MAX_CONCURRENCY="${BALLSIM_API_MAX_CONCURRENCY:-4}"

cleanup() {
    if [[ -n "${BALLSIM_API_PID:-}" ]]; then
        kill "${BALLSIM_API_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

uv run --active python -m uvicorn open_instruct.code_utils.ballsim_api:app --host 0.0.0.0 --port 2345 >/tmp/ballsim_api.log 2>&1 &
BALLSIM_API_PID=$!

for _ in {1..120}; do
    if curl -sf "http://localhost:2345/health" >/dev/null; then
        break
    fi
    sleep 1
done
curl -sf "http://localhost:2345/health" >/dev/null

uv run --active python open_instruct/grpo_fast.py \
    --dataset_mixer_list bouncingsim/bouncingsim-MULTIOBJ-basic 32 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list bouncingsim/bouncingsim-MULTIOBJ-basic 16 \
    --dataset_mixer_eval_list_splits test \
    --max_prompt_token_length 4096 \
    --response_length 4096 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 4 \
    --num_samples_per_prompt_rollout 4 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --learning_rate 3e-7 \
    --total_episodes 64 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 3 \
    --local_eval_every 4 \
    --vllm_sync_backend gloo \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --single_gpu_mode \
    --push_to_hub false \
    --system_prompt_override_file scripts/train/debug/ballsim_system_prompt.txt \
    --ballsim_api_url "${BALLSIM_API_URL}/test_program" \
    --ballsim_max_execution_time 1.0 \
    --ballsim_scoring_mode all_pass
