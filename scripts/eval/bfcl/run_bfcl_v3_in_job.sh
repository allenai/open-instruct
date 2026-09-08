#!/bin/bash
# In-job runner for BFCL v3 on an OLMoE3 (KDA MoE) HF checkpoint: build the serving venv,
# start vLLM as an OpenAI-compatible server, install BFCL in its own venv, generate and score.
# Launched by scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh; not meant to be run by hand.
#
# Runs inside akshitab/olmo-core-tch2110cu128-rma-2026-08-04, the image olmo-eval uses for this
# model family (scripts/train/debug/eval_olmoe3_kda.sh). Everything installed here mirrors that
# launcher's dependency list: vllm 0.19.1 with torch 2.10.0+cu128, olmo-core at the branch pin
# without its dependencies, and the OLMoE3 vLLM / transformers plugins from Weka, since eval jobs
# carry no GitHub credential for the private scaling-ladders repo.
set -euo pipefail

: "${CKPT:?HF checkpoint directory}"
: "${OUT_DIR:?output directory for BFCL results and scores}"
BFCL_MODEL_NAME="${BFCL_MODEL_NAME:-olmoe3-kda-sft-FC}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${BFCL_MODEL_NAME%-FC}}"
TEST_CATEGORY="${TEST_CATEGORY:-single_turn,multi_turn}"
NUM_THREADS="${NUM_THREADS:-32}"
TEMPERATURE="${TEMPERATURE:-0.001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# Default completion cap. BFCL's OpenAI handler sends no max_tokens, so without this every
# non-terminating response runs to the context limit (~41% of open-ended prompts on this family).
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
# olmo3, not qwen3: vLLM's qwen3 parser needs <think>/</think> to be single tokens and refuses to
# start otherwise ("could not locate think start/end tokens"). This tokenizer spells them as
# ordinary text, which the olmo3 parser handles, including a <think> supplied by the prompt.
REASONING_PARSER="${REASONING_PARSER:-olmo3}"
# Beaker jobs on one node share the host network: a fixed port would let this job's readiness
# check (and every BFCL request) reach a sibling job's vLLM. Pick an unused high port, bind to
# loopback, and treat the server as ready only when it reports *this* model name.
pick_free_port() {
    python3 - <<'PY'
import socket
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}
PORT="${PORT:-$(pick_free_port)}"
# Abort on a failed smoke request by default; set SMOKE_STRICT=0 to only warn.
SMOKE_STRICT="${SMOKE_STRICT:-1}"
SERVER_TIMEOUT_S="${SERVER_TIMEOUT_S:-2400}"
PLUGIN_DIR="${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-emo/ladders/olmoe3}"
OLMO_CORE_REF="${OLMO_CORE_REF:-f2cf93839}"
BFCL_EVAL_SPEC="${BFCL_EVAL_SPEC:-bfcl-eval}"
CLI_PY="${CLI_PY:?path to bfcl_cli_with_olmo_models.py}"

# The image ships CUDA 12.8 with ptxas under conda; Triton (used by the KDA kernels) must be
# pointed at it, as the olmo-eval launcher does.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/opt/conda/bin/ptxas}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "$OUT_DIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "checkpoint: $CKPT"
test -f "$CKPT/config.json" || { echo "no config.json in $CKPT (needs the HF export, not the DCP directory)" >&2; exit 2; }
test -d "$PLUGIN_DIR/vllm_plugin" || { echo "plugin dir $PLUGIN_DIR/vllm_plugin not found (is Weka mounted?)" >&2; exit 2; }

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    pip install -q uv
fi

# ---- serving venv -----------------------------------------------------------------------------
VLLM_VENV=/opt/venv-vllm
log "building serving venv at $VLLM_VENV"
uv venv -q "$VLLM_VENV" --python "$(command -v python3)"
VPIP=(uv pip install -q --python "$VLLM_VENV/bin/python")
# vllm pins its own torch (2.10.0); take it from the cu128 index to match the image's driver.
"${VPIP[@]}" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "vllm==0.19.1" "datasets==4.8.4"
# olmo-core --no-deps (its torch pin would fight vllm's), so its runtime deps are listed by hand;
# fla provides the KDA kernels.
"${VPIP[@]}" "cached-path>=1.7.2" "dataclass-extensions>=0.3.0" bettermap importlib_resources safetensors rich pandas "flash-linear-attention==0.4.1"
"${VPIP[@]}" --no-deps "ai2-olmo-core[transformers] @ git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_REF}"
"${VPIP[@]}" "$PLUGIN_DIR/vllm_plugin"
"${VPIP[@]}" --no-deps "$PLUGIN_DIR/transformers_plugin"
"$VLLM_VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
uv pip list --python "$VLLM_VENV/bin/python" 2>/dev/null | grep -iE "^(vllm|torch|transformers|flash|triton|ai2-olmo-core|olmoe3|fla) " || true

# ---- BFCL venv, separate so its pins never touch the serving stack ---------------------------
BFCL_VENV=/opt/venv-bfcl
log "building BFCL venv at $BFCL_VENV"
uv venv -q "$BFCL_VENV" --python "$(command -v python3)"
# soundfile: bfcl_eval imports qwen_agent at import time (for its Qwen API handler), and
# qwen_agent imports soundfile without declaring it. Import the CLI module here, before the
# model is loaded, so any further undeclared dependency fails in seconds rather than after a
# multi-minute model load.
uv pip install -q --python "$BFCL_VENV/bin/python" "$BFCL_EVAL_SPEC" soundfile
"$BFCL_VENV/bin/python" -c "import importlib.metadata as m; import bfcl_eval.__main__; print('bfcl-eval', m.version('bfcl-eval'), 'imports cleanly')"

# ---- serve ----------------------------------------------------------------------------------
# Flags mirror the olmo-eval provider kwargs for this family: eager mode (torch.compile does not
# handle the fla kernels), fp32 SSM cache, flash-attn backend. Tool calls come out in Qwen3-Coder
# style XML, which the Olmo 3.5 template adopted, so vLLM's qwen3_xml parser turns them into
# structured tool_calls; the olmo3 reasoning parser strips the <think> block the template forces.
log "starting vLLM on port $PORT"
"$VLLM_VENV/bin/vllm" serve "$CKPT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --enforce-eager \
    --mamba-ssm-cache-dtype float32 \
    --attention-backend FLASH_ATTN \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --reasoning-parser "$REASONING_PARSER" \
    --override-generation-config "{\"max_new_tokens\": $MAX_OUTPUT_TOKENS}" \
    > "$OUT_DIR/vllm_server.log" 2>&1 &
SERVER_PID=$!
trap 'log "stopping vLLM (pid $SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

# On failure, the API server's own traceback only says "engine core initialization failed"; the
# cause is in the EngineCore process's lines earlier in the log, so surface those specifically.
dump_server_failure() {
    echo "==== vLLM server log: error lines with context ====" >&2
    grep -nE "Error|Exception|Traceback|not supported|No module|CUDA out of memory|Killed" "$OUT_DIR/vllm_server.log" | grep -v "raise RuntimeError\|Engine core initialization failed" | head -40 >&2
    echo "==== vLLM server log: EngineCore lines ====" >&2
    grep -E "EngineCore" "$OUT_DIR/vllm_server.log" | tail -80 >&2
    echo "==== vLLM server log: last 40 lines ====" >&2
    tail -40 "$OUT_DIR/vllm_server.log" >&2
}
server_ready() {
    # Ready means our model id is listed, not merely that something answers on the port.
    curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"id\":\"$SERVED_MODEL_NAME\""
}
deadline=$((SECONDS + SERVER_TIMEOUT_S))
until server_ready; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "vLLM exited during startup" >&2; dump_server_failure; exit 3; fi
    if (( SECONDS > deadline )); then echo "vLLM not ready after ${SERVER_TIMEOUT_S}s" >&2; dump_server_failure; exit 3; fi
    sleep 10
done
log "vLLM ready on port $PORT: $(curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 300)"

# One end-to-end tool call before spending hours on the suite: proves the template renders
# tools, the model emits the XML, and the parser turns it into a structured tool_call.
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED_MODEL_NAME\", \"temperature\": 0,
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Paris in celsius?\"}],
  \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\",
     \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}}]
}" | tee "$OUT_DIR/smoke_tool_call.json" | head -c 1500; echo
if ! grep -q '"tool_calls"' "$OUT_DIR/smoke_tool_call.json"; then
    log "smoke request produced no structured tool_calls; check the parser and template before trusting scores"
    if [ "$SMOKE_STRICT" = "1" ]; then echo "aborting (SMOKE_STRICT=1)" >&2; exit 4; fi
fi

# ---- BFCL ------------------------------------------------------------------------------------
export BFCL_PROJECT_ROOT="$OUT_DIR"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=EMPTY
export BFCL_MODEL_NAME BFCL_SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
touch "$OUT_DIR/.env"  # BFCL loads PROJECT_ROOT/.env; keep it present but empty
BFCL=("$BFCL_VENV/bin/python" "$CLI_PY")

log "bfcl generate: model=$BFCL_MODEL_NAME categories=$TEST_CATEGORY threads=$NUM_THREADS"
"${BFCL[@]}" generate --model "$BFCL_MODEL_NAME" --test-category "$TEST_CATEGORY" \
    --num-threads "$NUM_THREADS" --temperature "$TEMPERATURE" --allow-overwrite 2>&1 | tee "$OUT_DIR/bfcl_generate.log"

log "bfcl evaluate"
"${BFCL[@]}" evaluate --model "$BFCL_MODEL_NAME" --test-category "$TEST_CATEGORY" 2>&1 | tee "$OUT_DIR/bfcl_evaluate.log"

log "scores:"
for f in data_overall.csv data_non_live.csv data_live.csv data_multi_turn.csv; do
    if [ -f "$OUT_DIR/score/$f" ]; then echo "== $f"; grep -iE "Rank|$BFCL_MODEL_NAME" "$OUT_DIR/score/$f" | cut -c1-400; fi
done
log "done; results under $OUT_DIR/result and $OUT_DIR/score"
