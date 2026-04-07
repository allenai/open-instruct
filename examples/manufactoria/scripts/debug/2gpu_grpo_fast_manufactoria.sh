#!/bin/bash

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"
export MANUFACTORIA_API_URL="${MANUFACTORIA_API_URL:-http://localhost:1235}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

cleanup() {
    if [[ -n "${MANUFACTORIA_API_PID:-}" ]]; then
        kill "${MANUFACTORIA_API_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

uv run --active python -m uvicorn examples.manufactoria.api:app --host 0.0.0.0 --port 1235 >/tmp/manufactoria_api.log 2>&1 &
MANUFACTORIA_API_PID=$!

for _ in {1..120}; do
    if curl -sf "http://localhost:1235/health" >/dev/null; then
        break
    fi
    sleep 1
done
curl -sf "http://localhost:1235/health" >/dev/null

uv run --active python -m examples.grpo_fast \
    --dataset_mixer_list manufactoria/basic_mix_train 8 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list manufactoria/basic_mix_test 4 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 512 \
    --pack_length 1536 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 2 \
    --num_samples_per_prompt_rollout 2 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --learning_rate 3e-7 \
    --total_episodes 8 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --async_steps 1 \
    --local_eval_every 1 \
    --gradient_checkpointing \
    --push_to_hub false \
    --verbose \
    --manufactoria_api_url "${MANUFACTORIA_API_URL}/test_solution" \
    --manufactoria_max_execution_time 1.0 \
    --manufactoria_scoring_mode all_pass
