#!/bin/bash
# In-job runner for BFCL v3 on an OLMoE3 (KDA MoE) HF checkpoint: build the serving venv,
# start vLLM as an OpenAI-compatible server, install BFCL in its own venv, generate and score.
# Launched by scripts/eval/bfcl/olmoe3_kda_bfcl_v3.sh; not meant to be run by hand.
#
# Runs inside akshitab/olmo-core-tch2110cu128-rma-2026-08-04, the image olmo-eval uses for this
# model family. Serving (venv build, vllm serve flags, readiness and smoke checks) comes from
# scripts/eval/olmoe3_kda_vllm_serve.sh, shared with the terminal-bench runner.
set -euo pipefail

: "${CKPT:?HF checkpoint directory}"
: "${OUT_DIR:?output directory for BFCL results and scores}"
BFCL_MODEL_NAME="${BFCL_MODEL_NAME:-olmoe3-kda-sft-FC}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${BFCL_MODEL_NAME%-FC}}"
TEST_CATEGORY="${TEST_CATEGORY:-single_turn,multi_turn}"
NUM_THREADS="${NUM_THREADS:-32}"
TEMPERATURE="${TEMPERATURE:-0.001}"
# Default completion cap. BFCL's OpenAI handler sends no max_tokens, so without this every
# non-terminating response runs to the context limit (~41% of open-ended prompts on this family).
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
BFCL_EVAL_SPEC="${BFCL_EVAL_SPEC:-bfcl-eval}"
CLI_PY="${CLI_PY:?path to bfcl_cli_with_olmo_models.py}"
SERVE_LIB="${SERVE_LIB:?path to olmoe3_kda_vllm_serve.sh}"
# Abort on a failed smoke request by default; set SMOKE_STRICT=0 to only warn.
SMOKE_STRICT="${SMOKE_STRICT:-1}"

# shellcheck source=scripts/eval/olmoe3_kda_vllm_serve.sh
source "$SERVE_LIB"
export VLLM_LOG="$OUT_DIR/vllm_server.log"
mkdir -p "$OUT_DIR"
log "checkpoint: $CKPT"
olmoe3_vllm_defaults

# ---- BFCL venv first, separate so its pins never touch the serving stack ----------------------
# soundfile: bfcl_eval imports qwen_agent at import time (for its Qwen API handler), and
# qwen_agent imports soundfile without declaring it. Import the CLI module here, before the
# model is loaded, so any further undeclared dependency fails in seconds rather than after a
# multi-minute model load.
if ! command -v uv >/dev/null 2>&1; then log "installing uv"; pip install -q uv; fi
BFCL_VENV=/opt/venv-bfcl
log "building BFCL venv at $BFCL_VENV"
uv venv -q --clear "$BFCL_VENV" --python "$(command -v python3)"  # --clear: idempotent if the container restarts
uv pip install -q --python "$BFCL_VENV/bin/python" "$BFCL_EVAL_SPEC" soundfile
"$BFCL_VENV/bin/python" -c "import importlib.metadata as m; import bfcl_eval.__main__; print('bfcl-eval', m.version('bfcl-eval'), 'imports cleanly')"

# ---- serve --------------------------------------------------------------------------------------
build_olmoe3_vllm_venv
start_olmoe3_vllm
trap 'stop_olmoe3_vllm' EXIT
wait_for_olmoe3_vllm
if ! smoke_olmoe3_tool_call "$OUT_DIR/smoke_tool_call.json"; then
    if [ "$SMOKE_STRICT" = "1" ]; then echo "aborting (SMOKE_STRICT=1)" >&2; exit 4; fi
fi

# ---- BFCL ------------------------------------------------------------------------------------
export BFCL_PROJECT_ROOT="$OUT_DIR"
export OPENAI_API_KEY=EMPTY
export BFCL_MODEL_NAME BFCL_SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
touch "$OUT_DIR/.env"  # BFCL loads PROJECT_ROOT/.env; keep it present but empty
BFCL=("$BFCL_VENV/bin/python" "$CLI_PY")

# Resume by default: BFCL loads the per-category result files already under OUT_DIR/result and
# generates only the missing test cases, so a rerun into the same RUN_NAME (after a container
# restart, or a relaunch) finishes what the previous job left instead of redoing the whole
# suite. BFCL_ALLOW_OVERWRITE=1 deletes them first for a clean regeneration.
GEN_FLAGS=()
if [ "${BFCL_ALLOW_OVERWRITE:-0}" = "1" ]; then GEN_FLAGS+=(--allow-overwrite); fi
RESULT_DIR="$OUT_DIR/result/$BFCL_MODEL_NAME"
# A job killed mid-write can leave a truncated last line, which BFCL's loader would choke on.
"$BFCL_VENV/bin/python" - "$RESULT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for f in (root.rglob("*.json") if root.is_dir() else []):
    lines = f.read_text().splitlines()
    keep = []
    for line in lines:
        try:
            json.loads(line); keep.append(line)
        except ValueError:
            print(f"dropping unparsable result line in {f}")
    if len(keep) != len(lines):
        f.write_text("".join(l + "\n" for l in keep))
PY
existing=0
if [ -d "$RESULT_DIR" ]; then existing=$(find "$RESULT_DIR" -name '*.json' -exec cat {} + | wc -l | tr -d ' '); fi
log "bfcl generate: model=$BFCL_MODEL_NAME categories=$TEST_CATEGORY threads=$NUM_THREADS existing_results=$existing overwrite=${BFCL_ALLOW_OVERWRITE:-0}"
"${BFCL[@]}" generate --model "$BFCL_MODEL_NAME" --test-category "$TEST_CATEGORY" \
    --num-threads "$NUM_THREADS" --temperature "$TEMPERATURE" "${GEN_FLAGS[@]}" 2>&1 | tee -a "$OUT_DIR/bfcl_generate.log"

log "bfcl evaluate"
"${BFCL[@]}" evaluate --model "$BFCL_MODEL_NAME" --test-category "$TEST_CATEGORY" 2>&1 | tee "$OUT_DIR/bfcl_evaluate.log"

log "scores:"
for f in data_overall.csv data_non_live.csv data_live.csv data_multi_turn.csv; do
    if [ -f "$OUT_DIR/score/$f" ]; then echo "== $f"; grep -iE "Rank|$BFCL_MODEL_NAME" "$OUT_DIR/score/$f" | cut -c1-400; fi
done
log "done; results under $OUT_DIR/result and $OUT_DIR/score"
