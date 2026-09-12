#!/bin/bash
# Serve an OLMoE3 (KDA MoE, HF `olmo3moe`) checkpoint with vLLM as an OpenAI-compatible endpoint.
# Sourced by the in-job eval runners under scripts/eval/; every function writes progress with
# `log` and fails loudly, since the callers run under `set -e`.
#
# The architecture is not in stock vLLM or transformers. It loads through the scaling-ladders
# plugins on exactly the stack olmo-eval uses for this family (scripts/train/debug/
# eval_olmoe3_kda.sh): vllm 0.19.1 with torch 2.10.0+cu128, olmo-core at the branch pin without
# its dependencies, flash-linear-attention for the KDA kernels, and the two plugins installed
# from Weka because eval jobs carry no GitHub credential for the private scaling-ladders repo.
#
# Verified end to end on ceres H100s (BFCL eval, 2026-09-08): model load ~2 min, a tool-calling
# request returns structured tool_calls with the reasoning separated into its own field.
#
# Inputs (environment, all optional except CKPT):
#   CKPT                    HF export directory (must contain config.json)
#   SERVED_MODEL_NAME       model id the server answers to (default: basename of CKPT)
#   PORT                    default: an unused loopback port -- jobs on one Beaker node share the
#                           host network, so a fixed port can reach a sibling job's server
#   MAX_MODEL_LEN           default 65536, the family's native window
#   MAX_OUTPUT_TOKENS       default completion cap when a client sends no max_tokens (8192)
#   GPU_MEMORY_UTILIZATION  default 0.9
#   TENSOR_PARALLEL         default 1; the plugin's TP support is untested
#   DATA_PARALLEL           default 1; extra replicas on extra GPUs, the safe way to add throughput
#   TOOL_CALL_PARSER        default qwen3_xml: the Olmo 3.5 template emits Qwen3-Coder-style XML
#   REASONING_PARSER        default olmo3: string-based, handles <think>/</think> as plain text and
#                           a <think> supplied by the prompt (the qwen3 parser needs single tokens)
#   SERVER_TIMEOUT_S        readiness deadline, default 2400
#   PLUGIN_DIR, OLMO_CORE_REF, VLLM_VENV, VLLM_LOG
#
# Outputs: PORT, SERVER_PID, OPENAI_BASE_URL exported; `stop_olmoe3_vllm` for cleanup.

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

pick_free_port() {
    python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

olmoe3_vllm_defaults() {
    : "${CKPT:?HF checkpoint directory}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$CKPT")}"
    PORT="${PORT:-$(pick_free_port)}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
    MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
    TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
    DATA_PARALLEL="${DATA_PARALLEL:-1}"
    TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
    REASONING_PARSER="${REASONING_PARSER:-olmo3}"
    SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-2400}"
    PLUGIN_DIR="${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-emo/ladders/olmoe3}"
    OLMO_CORE_REF="${OLMO_CORE_REF:-f2cf93839}"
    VLLM_VENV="${VLLM_VENV:-/opt/venv-vllm}"
    VLLM_LOG="${VLLM_LOG:-/tmp/vllm_server.log}"
    # Triton (used by the KDA kernels) needs a ptxas matching the CUDA runtime. Prefer a conda
    # one where the image has it; otherwise the variable must be *unset*, not empty: the
    # olmo-core image exports it empty, and triton 3.6 treats any non-None value as a path,
    # so an empty one dies in the first autotune with "PermissionError: Permission denied: ''"
    # instead of falling back to the bundled binary.
    if [ -z "${TRITON_PTXAS_PATH:-}" ]; then
        if [ -x /opt/conda/bin/ptxas ]; then export TRITON_PTXAS_PATH=/opt/conda/bin/ptxas; else unset TRITON_PTXAS_PATH; fi
    fi
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    # vLLM gives its engine cores 600 s to come up by default; loading 35 GB of weights from
    # Weka under contention has exceeded that (TBLite job 01M29RXQEJQXJ7PJYQJTNGHM3P). Give
    # the cores the same budget as our own readiness wait.
    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$SERVER_TIMEOUT_S}"
    export PORT SERVED_MODEL_NAME
    export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"

    test -f "$CKPT/config.json" || { echo "no config.json in $CKPT (needs the HF export, not the DCP directory)" >&2; return 2; }
    test -d "$PLUGIN_DIR/vllm_plugin" || { echo "plugin dir $PLUGIN_DIR/vllm_plugin not found (is Weka mounted?)" >&2; return 2; }
}

build_olmoe3_vllm_venv() {
    if ! command -v uv >/dev/null 2>&1; then
        log "installing uv"
        pip install -q uv 2>/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    fi
    log "building serving venv at $VLLM_VENV"
    uv venv -q --clear "$VLLM_VENV" --python "$(command -v python3)"  # --clear: idempotent if the container restarts
    local vpip=(uv pip install -q --python "$VLLM_VENV/bin/python")
    # vllm pins its own torch (2.10.0); take it from the cu128 index to match the driver stack.
    # transformers is pinned to what the verified runs resolved (2026-09-08); left free, the
    # same install started pulling 5.17.0 two days later, and a different tokenizer / template
    # stack would make later evals incomparable with the earlier ones.
    "${vpip[@]}" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "vllm==0.19.1" "datasets==4.8.4" "transformers==4.57.6"
    # olmo-core --no-deps (its torch pin would fight vllm's), so its runtime deps are listed by
    # hand; fla provides the KDA kernels.
    "${vpip[@]}" "cached-path>=1.7.2" "dataclass-extensions>=0.3.0" bettermap importlib_resources safetensors rich pandas "flash-linear-attention==0.4.1"
    "${vpip[@]}" --no-deps "ai2-olmo-core[transformers] @ git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_REF}"
    "${vpip[@]}" "$PLUGIN_DIR/vllm_plugin"
    "${vpip[@]}" --no-deps "$PLUGIN_DIR/transformers_plugin"
    "$VLLM_VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
    uv pip list --python "$VLLM_VENV/bin/python" 2>/dev/null | grep -iE "^(vllm|torch|transformers|flash|triton|ai2-olmo-core|olmoe3|fla) " || true
}

start_olmoe3_vllm() {
    # Flags mirror the olmo-eval provider kwargs for this family: eager mode (torch.compile does
    # not handle the fla kernels), fp32 SSM cache, flash-attn backend.
    log "starting vLLM on 127.0.0.1:$PORT as $SERVED_MODEL_NAME (log: $VLLM_LOG)"
    "$VLLM_VENV/bin/vllm" serve "$CKPT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host 127.0.0.1 --port "$PORT" \
        --trust-remote-code \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --data-parallel-size "$DATA_PARALLEL" \
        --enforce-eager \
        --mamba-ssm-cache-dtype float32 \
        --attention-backend FLASH_ATTN \
        --enable-auto-tool-choice \
        --tool-call-parser "$TOOL_CALL_PARSER" \
        --reasoning-parser "$REASONING_PARSER" \
        --override-generation-config "{\"max_new_tokens\": $MAX_OUTPUT_TOKENS}" \
        ${VLLM_EXTRA_ARGS:-} \
        > "$VLLM_LOG" 2>&1 &
    SERVER_PID=$!
    export SERVER_PID
}

stop_olmoe3_vllm() {
    if [ -n "${SERVER_PID:-}" ]; then log "stopping vLLM (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null || true; fi
}

dump_olmoe3_vllm_failure() {
    # The API server's traceback only says "engine core initialization failed"; the cause is in
    # the EngineCore process's lines earlier in the log.
    echo "==== vLLM server log: error lines with context ====" >&2
    grep -nE "Error|Exception|Traceback|not supported|No module|CUDA out of memory|Killed" "$VLLM_LOG" | grep -v "raise RuntimeError\|Engine core initialization failed" | head -40 >&2
    echo "==== vLLM server log: EngineCore lines ====" >&2
    grep -E "EngineCore" "$VLLM_LOG" | tail -80 >&2
    echo "==== vLLM server log: last 40 lines ====" >&2
    tail -40 "$VLLM_LOG" >&2
}

olmoe3_vllm_ready() {
    # Ready means our model id is listed, not merely that something answers on the port.
    curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"id\":\"$SERVED_MODEL_NAME\""
}

wait_for_olmoe3_vllm() {
    local deadline=$((SECONDS + SERVER_TIMEOUT_S))
    until olmoe3_vllm_ready; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "vLLM exited during startup" >&2; dump_olmoe3_vllm_failure; return 3; fi
        if (( SECONDS > deadline )); then echo "vLLM not ready after ${SERVER_TIMEOUT_S}s" >&2; dump_olmoe3_vllm_failure; return 3; fi
        sleep 10
    done
    log "vLLM ready on port $PORT: $(curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300)"
}

smoke_olmoe3_tool_call() {
    # One end-to-end tool call before spending hours on a suite: proves the template renders
    # tools, the model emits the XML, and the parser turns it into a structured tool_call.
    local out="${1:-/tmp/smoke_tool_call.json}"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\": \"$SERVED_MODEL_NAME\", \"temperature\": 0,
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Paris in celsius?\"}],
      \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\",
         \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}}]
    }" | tee "$out" | head -c 1500; echo
    if ! grep -q '"tool_calls"' "$out"; then
        echo "smoke request produced no structured tool_calls; check the parser and template before trusting scores" >&2
        return 4
    fi
    log "smoke tool call OK"
}
