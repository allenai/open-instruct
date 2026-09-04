#!/usr/bin/env bash
#
# Inner script for the multi-model thinking-trace sweep. Serves each model in
# MODELS with vLLM in turn, generates traces against it, tears it down, and
# moves on. One Beaker job covers the whole slate.
#
# Why one job rather than one per model: nothing outside the job advances the
# sequence, so the sweep keeps running unattended. The costs that buys back:
#
#   * Per-model isolation. A model that fails to serve or generate is logged and
#     skipped; the sweep continues with the next one.
#   * Idempotent resume. A model whose traces are already complete in the weka
#     store is skipped outright, so an auto-resumed job picks up where the
#     previous attempt stopped instead of redoing finished work.
#   * Periodic sync. Traces stream to /results and weka while generating, so a
#     crash costs at most SYNC_INTERVAL of work rather than the whole model.
#
# Env vars (set by beaker_configs/launch_thinking_traces_sweep.sh):
#   MODELS             space-separated HF repo ids, served in order
#   VLLM_VERSION       default 0.28.0 (needs >= 0.28 for GLM-5.2/Kimi-K2.x/
#                      DeepSeek-V3.2/Qwen3.5 architectures)
#   GPU_COUNT/TP_SIZE  GPUs and tensor-parallel size (default: 4 / GPU_COUNT)
#   MAX_MODEL_LEN      context (default 131072)
#   MAX_TOKENS         per-completion cap (default 128000)
#   NUM_PROMPTS        prompts per model (default 1000)
#   NUM_SAMPLES        completions per prompt (default 8)
#   CONCURRENCY        in-flight requests (default 256)
#   HF_REPO_ID         optional dataset repo to push each model's traces to
#   TRACE_STORE        weka dir for durable traces + resume markers
#   SYNC_INTERVAL      seconds between syncs (default 180)

set -uo pipefail   # NOT -e: a failing model must not kill the sweep

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${MODELS:?set MODELS}"
: "${VLLM_VERSION:=0.28.0}"
: "${SERVE_PORT:=8008}"
: "${GPU_COUNT:=4}"
: "${TP_SIZE:=$GPU_COUNT}"
: "${MAX_MODEL_LEN:=131072}"
: "${MAX_TOKENS:=128000}"
: "${MAX_PROMPT_TOKENS:=1536}"
: "${NUM_PROMPTS:=1000}"
: "${NUM_SAMPLES:=8}"
: "${TEMPERATURE:=0.6}"
: "${TOP_P:=0.95}"
: "${SEED:=1234}"
: "${CONCURRENCY:=256}"
: "${VLLM_MAX_NUM_SEQS:=$CONCURRENCY}"
: "${VLLM_READY_TIMEOUT:=5400}"
: "${DATASET:=allenai/Dolci-Think-SFT-7B}"
: "${RESULTS_DIR:=/results}"
: "${SYNC_INTERVAL:=180}"
: "${HF_REPO_ID:=}"

# vLLM reserves VLLM_PORT as the base of its internal port range; leaking one in
# makes every parallel rank derive the same rendezvous port (EADDRINUSE).
unset VLLM_PORT

REPO_ROOT="$(pwd)"
mkdir -p "$RESULTS_DIR"

if [ -z "${TRACE_STORE:-}" ] && [ -d /weka/oe-adapt-default ]; then
    TRACE_STORE="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/thinking_traces"
fi
: "${TRACE_STORE:=/tmp/thinking_traces}"
mkdir -p "$TRACE_STORE"

if [ -z "${HF_CACHE_DIR:-}" ] && [ -d /weka/oe-adapt-default ]; then
    HF_CACHE_DIR="/weka/oe-adapt-default/${BEAKER_USER_ID:-shared}/hf_cache"
fi
if [ -n "${HF_CACHE_DIR:-}" ]; then
    mkdir -p "$HF_CACHE_DIR"; export HF_HOME="$HF_CACHE_DIR"
fi

log "sweep configuration"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
cat <<EOF
  models        : ${MODELS}
  vLLM          : ${VLLM_VERSION}   TP=${TP_SIZE} over ${GPU_COUNT} GPUs
  context       : max_model_len=${MAX_MODEL_LEN}  max_tokens=${MAX_TOKENS}
  sampling      : ${NUM_PROMPTS} prompts x ${NUM_SAMPLES} samples, T=${TEMPERATURE} top_p=${TOP_P} seed=${SEED}
  concurrency   : ${CONCURRENCY}
  trace store   : ${TRACE_STORE}
  HF_HOME       : ${HF_HOME:-<default>}
  hub repo      : ${HF_REPO_ID:-<none: push skipped>}
EOF

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

UV_RUN=( uv run --no-project --python 3.11
         --with datasets --with transformers --with openai --with numpy --with huggingface_hub )
log "pre-resolving client deps"
"${UV_RUN[@]}" python -c "import datasets, transformers, openai, numpy, huggingface_hub; print('client deps ready')" \
    || { log "FATAL: client dependency resolution failed"; exit 1; }

# --- per-model run ------------------------------------------------------------
run_one_model() {
    local model="$1"
    local served; served="$(basename "$model" | tr '[:upper:]' '[:lower:]')"
    local traces="$RESULTS_DIR/traces_${served}.jsonl"
    local store="$TRACE_STORE/traces_${served}.jsonl"
    local done_marker="$store.done"

    if [ -f "$done_marker" ]; then
        log "SKIP ${model}: already complete ($(wc -l < "$store" 2>/dev/null || echo 0) traces in $store)"
        cp "$store" "$traces" 2>/dev/null || true
        return 0
    fi

    log "MODEL ${model} -> served as ${served}"
    local vllm_log=/tmp/vllm_${served}.log

    uvx "vllm==${VLLM_VERSION}" serve "$model" \
        --served-model-name "$served" \
        --port "$SERVE_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --enable-prefix-caching \
        --trust-remote-code \
        >"$vllm_log" 2>&1 &
    local vllm_pid=$!

    log "waiting for vllm on :$SERVE_PORT (up to ${VLLM_READY_TIMEOUT}s; weights are 400-760 GB on a cold cache)"
    local deadline=$(( SECONDS + VLLM_READY_TIMEOUT )) ticks=0
    until curl -sf "http://localhost:$SERVE_PORT/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            log "FAILED ${model}: vllm died. tail:"; tail -60 "$vllm_log"
            cp "$vllm_log" "$RESULTS_DIR/" 2>/dev/null || true
            return 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            log "FAILED ${model}: vllm not ready in ${VLLM_READY_TIMEOUT}s. tail:"; tail -60 "$vllm_log"
            kill "$vllm_pid" 2>/dev/null; cp "$vllm_log" "$RESULTS_DIR/" 2>/dev/null || true
            return 1
        fi
        ticks=$(( ticks + 1 ))
        if [ $(( ticks % 12 )) = 0 ]; then
            log "still loading ${served} (${SECONDS}s):"; tail -3 "$vllm_log" 2>/dev/null || true
        fi
        sleep 10
    done
    log "vllm ready for ${served} after ${SECONDS}s"

    # stream partial traces out while generating
    ( while true; do sleep "$SYNC_INTERVAL"
        cp "$traces" "$store" 2>/dev/null || true
        [ -f "$traces" ] && log "sync: $(wc -l < "$traces") traces for ${served}"
      done ) &
    local sync_pid=$!

    PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/generate_traces.py \
        --model "$served" --tokenizer "$model" \
        --api-base "http://localhost:${SERVE_PORT}/v1" \
        --dataset "$DATASET" \
        --num-prompts "$NUM_PROMPTS" --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --seed "$SEED" --concurrency "$CONCURRENCY" \
        --prompts-output "$RESULTS_DIR/prompts_${served}.jsonl" \
        --output "$traces" 2>&1 | tee "$RESULTS_DIR/generate_${served}.log"
    local rc=${PIPESTATUS[0]}

    kill "$sync_pid" 2>/dev/null; wait "$sync_pid" 2>/dev/null
    kill "$vllm_pid" 2>/dev/null; wait "$vllm_pid" 2>/dev/null
    tail -200 "$vllm_log" > "$RESULTS_DIR/vllm_tail_${served}.log" 2>/dev/null || true

    if [ "$rc" != "0" ]; then
        log "FAILED ${model}: generation exited $rc (partial traces kept)"
        cp "$traces" "$store" 2>/dev/null || true
        return 1
    fi

    cp "$traces" "$store" && touch "$done_marker"
    log "DONE ${model}: $(wc -l < "$traces") traces"

    PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/analyze_traces.py \
        --traces "${served}=${traces}" \
        --json-output "$RESULTS_DIR/summary_${served}.json" 2>&1 \
        | tee "$RESULTS_DIR/summary_${served}.txt"

    if [ -n "$HF_REPO_ID" ]; then
        log "pushing ${served} to ${HF_REPO_ID} (best effort)"
        PYTHONPATH="$REPO_ROOT" "${UV_RUN[@]}" python scripts/thinking_traces/push_traces_to_hub.py \
            --traces "$traces" --repo-id "$HF_REPO_ID" --config-name "$served" --best-effort 2>&1 | tail -20
    fi
    return 0
}

# --- sweep --------------------------------------------------------------------
SUCCEEDED=(); FAILED=()
for model in $MODELS; do
    if run_one_model "$model"; then SUCCEEDED+=("$model"); else FAILED+=("$model"); fi
    log "progress: ${#SUCCEEDED[@]} succeeded, ${#FAILED[@]} failed, of $(echo $MODELS | wc -w)"
done

log "sweep complete"
echo "  succeeded: ${SUCCEEDED[*]:-none}"
echo "  failed   : ${FAILED[*]:-none}"
ls -la "$RESULTS_DIR"
[ ${#SUCCEEDED[@]} -gt 0 ] || exit 1
exit 0
