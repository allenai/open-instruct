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

SCORE_MODE=pass_rate
EXP_NAME="${EXP_NAME:-qwen3_4b_it_manufac_${SCORE_MODE}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

cleanup() {
    if [[ -n "${MANUFACTORIA_API_PID:-}" ]]; then
        kill "${MANUFACTORIA_API_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

uv run --active python -m uvicorn open_instruct.code_utils.manufactoria_api:app --host 0.0.0.0 --port 1235 >/tmp/manufactoria_api.log 2>&1 &
MANUFACTORIA_API_PID=$!

for _ in {1..120}; do
    if curl -sf "http://localhost:1235/health" >/dev/null; then
        break
    fi
    sleep 1
done
curl -sf "http://localhost:1235/health" >/dev/null

uv run --active open_instruct/grpo_fast.py \
--run_name "${RUN_NAME}" \
--exp_name "${EXP_NAME}" \
--beta 0.0 \
--eval_pass_at_k 8 \
--load_ref_policy false \
--num_unique_prompts_rollout 48 \
--num_samples_per_prompt_rollout 16 \
--num_mini_batches 1 \
--num_epochs 1 \
--learning_rate 5e-7 \
--lr_scheduler_type constant \
--per_device_train_batch_size 1 \
--dataset_mixer_list mnoukhov/manufactoria-qwen3-4b-instruct-pass128 1.0 \
--dataset_mixer_list_splits train \
--dataset_mixer_eval_list mnoukhov/manufactoria-qwen3-4b-instruct-pass128 4 \
--dataset_mixer_eval_list_splits test \
--max_prompt_token_length 2048 \
--response_length 8192 \
--pack_length 10240 \
--model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
--apply_verifiable_reward true \
--manufactoria_api_url "http://localhost:1235/test_solution" \
--manufactoria_scoring_mode "${SCORE_MODE}" \
--temperature 1.0 \
--total_episodes 768000 \
--deepspeed_stage 2 \
--num_learners_per_node 1 \
--vllm_num_engines 1 \
--clip_higher 0.28 \
--seed 1 \
--local_eval_every 10 \
--save_freq 10 \
--checkpoint_state_freq 10 \
--gradient_checkpointing \
--with_tracking \
--push_to_hub false \
"$@"
