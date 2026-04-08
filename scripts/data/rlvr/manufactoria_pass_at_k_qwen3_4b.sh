#!/usr/bin/env bash
# Generate pass@32 datasets for manufactoria/has_train and manufactoria/has_test using Qwen3-4B-Instruct-2507,
# aligned with scripts/train/manufactoria/qwen3_4b_phase1_has_8gpu.sh (temperature, response length, Manufactoria API, pass_rate).
#
# Each output row includes a `difficulty` column (length N, values 1–4 per test) derived from per-test solve rates
# across the k samples—no separate pass-rate bucket datasets/splits.
#
# Requires Manufactoria API (same as training). Example local:
#   source configs/beaker_configs/manufactoria_api_setup.sh   # or start API yourself
#   export MANUFACTORIA_API_URL=http://localhost:1235
#   bash scripts/data/rlvr/manufactoria_pass_at_k_qwen3_4b.sh
#
# Default push targets: unset → mnoukhov/manufactoria-has-{train,test}-qwen3-4b-instruct-pass32
# (override owner with HF_DATASETS_OWNER). Set PUSH_TRAIN_REPO= to empty string to skip train push only.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-from_model}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_ENGINES="${NUM_ENGINES:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
SEED="${SEED:-1}"
SCORE_MODE="${SCORE_MODE:-pass_rate}"
MAX_EXEC_TIME="${MAX_EXEC_TIME:-1.0}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/tmp/manufactoria_pass_at_k_outputs}"
LIMIT="${LIMIT:-}"

TRAIN_SOURCE="${TRAIN_SOURCE:-manufactoria/has_train}"
TEST_SOURCE="${TEST_SOURCE:-manufactoria/has_test}"
SPLIT="${SPLIT:-train}"

HF_DATASETS_OWNER="${HF_DATASETS_OWNER:-mnoukhov}"
# Use ${var-word} (not :-) so PUSH_*="" explicitly skips push for that split.
PUSH_TRAIN_REPO="${PUSH_TRAIN_REPO-${HF_DATASETS_OWNER}/manufactoria-has-train-qwen3-4b-instruct-pass32}"
PUSH_TEST_REPO="${PUSH_TEST_REPO-${HF_DATASETS_OWNER}/manufactoria-has-test-qwen3-4b-instruct-pass32}"
PRIVATE_FLAG="${PRIVATE_FLAG:-}"

_extra_limit=()
if [[ -n "${LIMIT}" ]]; then
  _extra_limit=(--limit "${LIMIT}")
fi

_private=()
if [[ "${PRIVATE_FLAG}" == "1" || "${PRIVATE_FLAG}" == "true" ]]; then
  _private=(--private)
fi

_run_one() {
  local dataset="$1"
  local push_repo="$2"
  local push_args=()
  if [[ -n "${push_repo}" ]]; then
    push_args=(--push-to-hub "${push_repo}")
  fi

  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \
    --dataset "${dataset}" \
    --split "${SPLIT}" \
    --model "${MODEL_NAME}" \
    --chat-template "${CHAT_TEMPLATE}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --num-engines "${NUM_ENGINES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${SEED}" \
    --manufactoria-scoring-mode "${SCORE_MODE}" \
    --manufactoria-max-execution-time "${MAX_EXEC_TIME}" \
    --save-local-dir "${SAVE_LOCAL_DIR}" \
    "${_extra_limit[@]}" \
    "${_private[@]}" \
    "${push_args[@]}"
}

_run_one "${TRAIN_SOURCE}" "${PUSH_TRAIN_REPO}"
_run_one "${TEST_SOURCE}" "${PUSH_TEST_REPO}"
