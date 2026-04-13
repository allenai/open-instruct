#!/usr/bin/env bash
# Generate pass@k datasets for the train/test splits of one Manufactoria dataset repo.
#
# Requires Manufactoria API (same as training). Example local:
#   source configs/beaker_configs/manufactoria_api_setup.sh
#   bash scripts/data/rlvr/manufactoria_pass_at_k_qwen3_4b.sh
#
# By default, this runs both train and test from DATASET_NAME and skips push unless
# PUSH_DATASET_REPO is set.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/qwen3_4b_it_manufac_pass_rate__1__1776024283_checkpoints/step_650}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-from_model}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}"
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

DATASET_NAME="${DATASET_NAME:-mnoukhov/manufactoria-qwen3-4b-instruct-pass128}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"
PUSH_DATASET_REPO="${PUSH_DATASET_REPO-}"
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
  local split="$1"
  local push_args=()
  if [[ -n "${PUSH_DATASET_REPO}" ]]; then
    push_args=(--push-to-hub "${PUSH_DATASET_REPO}")
  fi

  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \
    --dataset "${DATASET_NAME}" \
    --split "${split}" \
    --model "${MODEL_NAME}" \
    --chat-template "${CHAT_TEMPLATE}" \
    --num-samples "${NUM_SAMPLES}" \
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
    --response_length "${RESPONSE_LENGTH}" \
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

_run_one "${TRAIN_SPLIT}"
_run_one "${TEST_SPLIT}"
