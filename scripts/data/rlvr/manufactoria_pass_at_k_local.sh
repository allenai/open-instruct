#!/usr/bin/env bash
# Local Manufactoria pass@k generation: starts the Manufactoria API, then runs the same
# train/test passes as manufactoria_pass_at_k_qwen3_4b.sh with that script's default
# hyperparameters inlined (override by editing this file or running the non-local script).

set -euo pipefail

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

export MANUFACTORIA_API_URL=http://localhost:1235

_run_one() {
  local split="$1"
  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \
    --dataset "mnoukhov/manufactoria-qwen3-4b-instruct-pass128" \
    --split "${split}" \
    --model "/weka/oe-adapt-default/allennlp/deletable_checkpoint/michaeln/qwen3_4b_it_manufac_pass_rate__1__1776024283_checkpoints/step_650" \
    --chat-template "from_model" \
    --num-samples 128 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --temperature 1.0 \
    --top-p 1.0 \
    --tensor-parallel-size 1 \
    --num-engines 8 \
    --gpu-memory-utilization 0.9 \
    --seed 1 \
    --manufactoria-scoring-mode "pass_rate" \
    --manufactoria-max-execution-time 1.0 \
    --save-local-dir "/tmp/manufactoria_pass_at_k_outputs" $@
}

_run_one "train[:1]"
# _run_one "test"
