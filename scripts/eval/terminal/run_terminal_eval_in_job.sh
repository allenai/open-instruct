#!/bin/bash
# In-job runner: Terminal-Bench 2.1 / 2.0 or OpenThoughts-TBLite on an OLMoE3 (KDA MoE) HF
# checkpoint, served with vLLM and driven by the tmax harbor pipeline.
# Launched by scripts/eval/terminal/olmoe3_kda_terminal_eval.sh; not meant to be run by hand.
#
# The evaluation itself (podman + patched harbor, the Vanillux2Agent bash agent, per-trial
# stats, copying results to Weka) is tmax's scripts/beaker/run_eval_in_job.sh, run unchanged
# except for two one-line substitutions applied at runtime, the same way that script patches
# harbor:
#
#   1. Its `uvx vllm==... serve` launch is replaced by a placeholder process. Stock vLLM cannot
#      load this architecture; we start vLLM ourselves beforehand through the scaling-ladders
#      plugin stack (scripts/eval/olmoe3_kda_vllm_serve.sh) on the port tmax's script then
#      probes, with the qwen3_xml tool parser and the olmo3 reasoning parser the Olmo 3.5
#      template needs.
#   2. Its hard-coded `--dataset "$DATASET"` becomes `$HARBOR_DATASET_FLAGS`, so a dataset can
#      also be given as a local task directory (`-p`). Terminal-Bench 2.1 is not in the harbor
#      registry that tmax's harbor pin (0.6.6) reads; it is a git repo of harbor tasks, which
#      we clone and pass with -p.
#
# Both substitutions must match exactly once or the run aborts, so drift in tmax shows up as
# a loud failure rather than a silently different evaluation.
#
# Inputs (environment):
#   CKPT                HF export directory                              (required)
#   SERVE_LIB           path to olmoe3_kda_vllm_serve.sh                 (required)
#   DATASET             harbor registry id, e.g. openthoughts-tblite@2.0 or terminal-bench@2.0;
#                       ignored when DATASET_GIT_URL is set
#   DATASET_GIT_URL     git repo of harbor tasks (e.g. Terminal-Bench 2.1); DATASET_GIT_REF pins
#                       it; DATASET_SUBDIR is the tasks directory inside (default tasks)
#   JOB_NAME, RESULTS_DIR, SERVED_MODEL_NAME, N_CONCURRENT, N_ATTEMPTS, N_TASKS,
#   AGENT_IMPORT_PATH, HARBOR_* passthroughs           -- as in tmax's launcher
#   TMAX_GIT_URL, TMAX_GIT_REF                          -- the tmax checkout to use
#   MAX_MODEL_LEN, MAX_OUTPUT_TOKENS, TENSOR_PARALLEL, GPU_MEMORY_UTILIZATION -- serving
set -euo pipefail

: "${CKPT:?HF checkpoint directory}"
: "${SERVE_LIB:?path to olmoe3_kda_vllm_serve.sh}"
: "${RESULTS_DIR:?results directory on Weka}"
TMAX_GIT_URL="${TMAX_GIT_URL:-https://github.com/shatu/tmax.git}"
TMAX_GIT_REF="${TMAX_GIT_REF:-f0a3db4792ccd6cf75c377ea7fe628c3b3ab9145}"  # pd_sft_regen, 2026-08-16
DATASET="${DATASET:-terminal-bench@2.0}"
DATASET_SUBDIR="${DATASET_SUBDIR:-tasks}"
WORKDIR="${WORKDIR:-/workspace}"
# Terminal tasks are long: allow the agent's 16k completions rather than the 8k BFCL default.
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$CKPT")}"

# shellcheck source=scripts/eval/olmoe3_kda_vllm_serve.sh
source "$SERVE_LIB"

log "checkpoint: $CKPT"
mkdir -p "$WORKDIR" "$RESULTS_DIR"

# ---- checkouts -------------------------------------------------------------------------------
if [ ! -d "$WORKDIR/tmax/.git" ]; then
    log "cloning tmax $TMAX_GIT_URL @ $TMAX_GIT_REF"
    git clone -q "$TMAX_GIT_URL" "$WORKDIR/tmax"
fi
git -C "$WORKDIR/tmax" checkout -q "$TMAX_GIT_REF"

if [ -n "${DATASET_GIT_URL:-}" ]; then
    if [ ! -d "$WORKDIR/dataset/.git" ]; then
        log "cloning dataset $DATASET_GIT_URL @ ${DATASET_GIT_REF:-HEAD}"
        git clone -q "$DATASET_GIT_URL" "$WORKDIR/dataset"
    fi
    [ -n "${DATASET_GIT_REF:-}" ] && git -C "$WORKDIR/dataset" checkout -q "$DATASET_GIT_REF"
    test -d "$WORKDIR/dataset/$DATASET_SUBDIR" || { echo "no $DATASET_SUBDIR/ in $DATASET_GIT_URL" >&2; exit 2; }
    HARBOR_DATASET_FLAGS="-p $WORKDIR/dataset/$DATASET_SUBDIR"
    log "dataset: local task directory $WORKDIR/dataset/$DATASET_SUBDIR ($(find "$WORKDIR/dataset/$DATASET_SUBDIR" -mindepth 1 -maxdepth 1 -type d | wc -l) tasks)"
else
    HARBOR_DATASET_FLAGS="--dataset $DATASET"
    log "dataset: harbor registry $DATASET"
fi
export HARBOR_DATASET_FLAGS

# ---- serve the model (before tmax's script, which then only probes the port) -----------------
export VLLM_LOG=/tmp/vllm.log   # tmax's script preserves this file into the job directory
olmoe3_vllm_defaults
build_olmoe3_vllm_venv
start_olmoe3_vllm
trap 'stop_olmoe3_vllm' EXIT
wait_for_olmoe3_vllm
smoke_olmoe3_tool_call "$RESULTS_DIR/smoke_tool_call.json"

# ---- patch tmax's runner: no second vLLM, dataset flags from the environment ------------------
RUNNER="$WORKDIR/tmax/scripts/beaker/run_eval_in_job.sh"
PATCHED="$WORKDIR/tmax/scripts/beaker/run_eval_in_job.olmoe3.sh"
python3 - "$RUNNER" "$PATCHED" <<'PY'
import pathlib, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = src.read_text()
subs = [
    # 1. keep VLLM_PID alive for the script's liveness check without starting a second server or
    #    truncating the log our own server is writing to.
    ('"${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &', 'sleep infinity &'),
    # 2. dataset source from the environment: registry id or -p <task dir>.
    ('--dataset "$DATASET"', '$HARBOR_DATASET_FLAGS'),
]
for old, new in subs:
    n = text.count(old)
    if n != 1:
        sys.exit(f"expected exactly one occurrence of {old!r} in {src}, found {n}; tmax's runner changed, refusing to guess")
    text = text.replace(old, new)
dst.write_text(text)
print(f"wrote {dst}")
PY

# ---- run tmax's pipeline against our server -------------------------------------------------
export MODEL_PATH="$CKPT" MODEL_REVISION=main
export VLLM_PORT="$PORT" TP_SIZE="${TENSOR_PARALLEL:-1}" DP_SIZE=1
export AGENT_IMPORT_PATH="${AGENT_IMPORT_PATH:-Vanillux2Agent:Vanillux2Agent}"
export JOB_NAME="${JOB_NAME:-${SERVED_MODEL_NAME}-${DATASET//[^A-Za-z0-9]/-}}"
export N_CONCURRENT="${N_CONCURRENT:-8}" N_ATTEMPTS="${N_ATTEMPTS:-1}"
log "handing over to tmax runner: job $JOB_NAME, agent $AGENT_IMPORT_PATH, $HARBOR_DATASET_FLAGS, port $PORT"
cd "$WORKDIR/tmax"
set +e
bash "$PATCHED"
rc=$?
set -e
log "tmax runner exited with $rc; results under $RESULTS_DIR/$JOB_NAME"
exit "$rc"
