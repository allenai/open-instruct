#!/usr/bin/env bash
#
# Inner script invoked inside a Beaker task. Serves one model with vLLM on the
# local GPUs, replays a seeded slice of a post-training SFT mixture through it,
# and records the token length of every <think>...</think> trace.
#
# Adapted from tmax's scripts/beaker/run_gen_solutions_in_job.sh. The shape is
# the same -- uvx vllm serve in the background, poll /v1/models, run a client
# against localhost, sync to /results -- but there is no podman and no task
# corpus here: the workload is a plain OpenAI-compatible chat client.
#
# Driven entirely by env vars (set by beaker_configs/launch_thinking_traces.sh):
#
#   -- vLLM serving --
#   VLLM_MODEL           HF repo to serve (required, e.g. Qwen/Qwen3-8B)
#   SERVED_MODEL_NAME    --served-model-name (required, e.g. qwen3-8b)
#   VLLM_VERSION         vLLM version for uvx (default: 0.23.0)
#   VLLM_PORT            port for vLLM (default: 8008)
#   GPU_COUNT            GPUs allocated to the job (default: 8)
#   TP_SIZE / DP_SIZE    parallelism (default: TP=1, DP=GPU_COUNT -- 8B models
#                        fit on one GPU, so N independent engines beat one
#                        sharded engine, especially on NVLink-less L40S)
#   MAX_MODEL_LEN        --max-model-len (default: 32768)
#   VLLM_GPU_UTIL        --gpu-memory-utilization (default: vllm's default)
#   VLLM_MAX_NUM_SEQS    --max-num-seqs (default: 64)
#   VLLM_EXTRA_ARGS      free-form extra args appended to vllm serve
#   VLLM_READY_TIMEOUT   seconds to wait for /v1/models (default: 1800)
#
#   -- sampling --
#   DATASET              HF dataset to draw prompts from
#                        (default: allenai/Dolci-Think-SFT-7B)
#   NUM_PROMPTS          prompts to sample (default: 200)
#   NUM_SAMPLES          completions per prompt (default: 4)
#   TEMPERATURE          default 0.6  (both vendors' recommended thinking temp)
#   TOP_P                default 0.95
#   MAX_TOKENS           per-completion cap (default: 30720). Traces that hit
#                        this are censored, so the script reports the rate.
#   MAX_PROMPT_TOKENS    skip prompts longer than this (default: 1536)
#   SEED                 sampling seed (default: 1234). Identical across models,
#                        which is what makes the two runs comparable.
#   CONCURRENCY          in-flight requests (default: 64)
#
#   -- plumbing --
#   RESULTS_DIR          where to write results (default: /results, persisted by
#                        gantry as the per-job result dataset)
#   SYNC_INTERVAL        seconds between progress logs (default: 120)
#   HF_CACHE_DIR         HF_HOME override (default: a weka path when mounted)

set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${VLLM_MODEL:?set VLLM_MODEL}"
: "${SERVED_MODEL_NAME:?set SERVED_MODEL_NAME}"
: "${VLLM_VERSION:=0.23.0}"
: "${VLLM_PORT:=8008}"
: "${GPU_COUNT:=8}"
: "${TP_SIZE:=1}"
: "${DP_SIZE:=$(( GPU_COUNT / TP_SIZE ))}"
: "${MAX_MODEL_LEN:=32768}"
: "${VLLM_MAX_NUM_SEQS:=64}"
: "${VLLM_READY_TIMEOUT:=1800}"
: "${DATASET:=allenai/Dolci-Think-SFT-7B}"
: "${NUM_PROMPTS:=200}"
: "${NUM_SAMPLES:=4}"
: "${TEMPERATURE:=0.6}"
: "${TOP_P:=0.95}"
: "${MAX_TOKENS:=30720}"
: "${MAX_PROMPT_TOKENS:=1536}"
: "${SEED:=1234}"
: "${CONCURRENCY:=64}"
: "${RESULTS_DIR:=/results}"
: "${SYNC_INTERVAL:=120}"

REPO_ROOT="$(pwd)"
mkdir -p "$RESULTS_DIR"

log "environment"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "repo root: $REPO_ROOT"

# --- 1. Base tooling ---------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# Keep the HF cache on weka when it is mounted so a second job on the same node
# does not re-download the weights.
if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/hf_cache"
fi
if [ -n "${HF_CACHE_DIR:-}" ]; then
    mkdir -p "$HF_CACHE_DIR"
    export HF_HOME="$HF_CACHE_DIR"
    log "HF_HOME=$HF_HOME"
fi

# --- 2. Start vLLM in the background -----------------------------------------
VLLM_LOG=/tmp/vllm.log
VLLM_LOG_TAIL_LINES="${VLLM_LOG_TAIL_LINES:-300}"

VLLM_CMD=( uvx "vllm==${VLLM_VERSION}" serve "$VLLM_MODEL"
           --served-model-name "$SERVED_MODEL_NAME"
           --port "$VLLM_PORT"
           --tensor-parallel-size "$TP_SIZE"
           --max-model-len "$MAX_MODEL_LEN"
           --max-num-seqs "$VLLM_MAX_NUM_SEQS"
           --enable-prefix-caching )
# NOTE: no request-logging flag. vLLM 0.23 renamed it to --enable-log-requests
# and defaults it off; the older --disable-log-requests is gone, and passing it
# aborts `vllm serve` with "unrecognized arguments".
if [ "$DP_SIZE" -gt 1 ]; then
    VLLM_CMD+=( --data-parallel-size "$DP_SIZE" )
fi
if [ -n "${VLLM_GPU_UTIL:-}" ]; then
    VLLM_CMD+=( --gpu-memory-utilization "$VLLM_GPU_UTIL" )
fi
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    VLLM_CMD+=( ${VLLM_EXTRA_ARGS} )
fi

# NOTE: no --reasoning-parser on purpose. Leaving it off keeps the literal
# </think> in the response content, so one parser handles both the Qwen3
# template (model emits its own <think>) and the R1-Distill template (which
# prefills <think>, so the completion starts inside the trace).

log "launching vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    log "cleanup: killing vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" > "$RESULTS_DIR/vllm_tail.log" 2>/dev/null || true
}
trap cleanup EXIT

# --- 3. Resolve client dependencies while the weights load --------------------
# --no-project: this repo's pyproject pulls in the full training stack, which we
# do not need (and cannot build quickly) just to drive an HTTP endpoint.
UV_RUN=( uv run --no-project --python 3.11
         --with "datasets" --with "transformers" --with "openai" --with "numpy" )

log "pre-resolving client deps"
"${UV_RUN[@]}" python -c "import datasets, transformers, openai, numpy; print('client deps ready')" &
DEPS_PID=$!

# --- 4. Wait for vLLM ---------------------------------------------------------
log "waiting for vllm on :$VLLM_PORT (up to ${VLLM_READY_TIMEOUT}s)"
_deadline=$(( SECONDS + VLLM_READY_TIMEOUT ))
until curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vllm process died — tail of $VLLM_LOG:"
        tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$_deadline" ]; then
        log "vllm not ready after ${VLLM_READY_TIMEOUT}s — tail of $VLLM_LOG:"
        tail -n "$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    sleep 10
done
log "vllm ready"
curl -s "http://localhost:$VLLM_PORT/v1/models" | head -c 600; echo

wait "$DEPS_PID" || { log "client dependency resolution failed"; exit 1; }

# --- 5. Generate traces -------------------------------------------------------
TRACES_JSONL="$RESULTS_DIR/traces_${SERVED_MODEL_NAME}.jsonl"
PROMPTS_JSONL="$RESULTS_DIR/prompts_${SERVED_MODEL_NAME}.jsonl"

log "generating traces -> $TRACES_JSONL"
PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/generate_traces.py \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$VLLM_MODEL" \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --dataset "$DATASET" \
    --num-prompts "$NUM_PROMPTS" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --seed "$SEED" \
    --concurrency "$CONCURRENCY" \
    --prompts-output "$PROMPTS_JSONL" \
    --output "$TRACES_JSONL" 2>&1 | tee "$RESULTS_DIR/generate.log"

# --- 6. Single-model summary --------------------------------------------------
# The cross-model comparison happens later, once both jobs' results are pulled
# down together; this is just so each job's own numbers are visible in its log.
log "summarizing $TRACES_JSONL"
PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/analyze_traces.py \
    --traces "${SERVED_MODEL_NAME}=${TRACES_JSONL}" \
    --json-output "$RESULTS_DIR/summary_${SERVED_MODEL_NAME}.json" 2>&1 \
    | tee "$RESULTS_DIR/summary_${SERVED_MODEL_NAME}.txt"

log "done: results in $RESULTS_DIR"
ls -la "$RESULTS_DIR"
