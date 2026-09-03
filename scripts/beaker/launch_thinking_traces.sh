#!/usr/bin/env bash
#
# Launch Beaker task(s) that measure thinking-trace length on a post-training
# mixture. Each task serves one model with vLLM on its own node and replays the
# same seeded prompt sample through it.
#
# Adapted from tmax's beaker_configs/launch_gen_solutions.sh: same Gantry
# submit-the-current-SHA approach and the same env-var contract with the inner
# script (scripts/beaker/run_thinking_traces_in_job.sh), without the podman
# base-image machinery, which that pipeline needed for agentic rollouts and
# this one does not.
#
# Both models must be launched with the same --seed, --num-prompts and
# --max-prompt-tokens: prompt selection is a deterministic function of those,
# which is what makes the two runs comparable. --both enforces this by
# construction.
#
# Usage:
#   ./scripts/beaker/launch_thinking_traces.sh [options]
#
# Examples:
#   # The default experiment: Qwen3-8B vs DeepSeek-R1-Distill-Llama-8B
#   ./scripts/beaker/launch_thinking_traces.sh --both
#
#   # Smoke test one model on a handful of prompts
#   ./scripts/beaker/launch_thinking_traces.sh \
#       --vllm-model Qwen/Qwen3-8B --num-prompts 8 --num-samples 2 --gpus 2
#
# The SHA must be pushed to the remote; local dirty changes are NOT included in
# the remote job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- defaults ----------------------------------------------------------------
LAUNCH_BOTH=0
VLLM_MODEL="Qwen/Qwen3-8B"
MODEL_B="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SERVED_MODEL_NAME=""
# vLLM 0.20+ pins torch 2.11, which is built against CUDA 13 and needs driver
# >= 580. ai2/neptune and ai2/jupiter run 570.x (CUDA 12.8), where it dies with
# "The NVIDIA driver on your system is too old (found version 12080)". 0.19.1 is
# the newest release still on torch 2.10 / CUDA 12 -- the same pairing as the
# ai2/cuda12.8-*-torch2.10.0 Beaker image. Only raise this for a cluster whose
# driver is new enough (e.g. the CUDA 13 B300 nodes).
VLLM_VERSION="0.19.1"
VLLM_PORT=8008
CLUSTER="ai2/neptune"
GPU_COUNT=8
TP_SIZE=1
DP_SIZE=""
MAX_MODEL_LEN=32768
VLLM_GPU_UTIL=""
VLLM_MAX_NUM_SEQS=64
VLLM_EXTRA_ARGS=""
DATASET="allenai/Dolci-Think-SFT-7B"
NUM_PROMPTS=200
NUM_SAMPLES=4
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=30720
MAX_PROMPT_TOKENS=1536
SEED=1234
CONCURRENCY=64
JOB_NAME=""
JOB_SUFFIX=""
PRIORITY="normal"
PREEMPTIBLE=0
TASK_TIMEOUT="6h"
BUDGET="ai2/oe-other"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/chrisg-onboarding}"
BEAKER_IMAGE="${BEAKER_IMAGE:-ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0}"
WEKA_MOUNT="oe-adapt-default:/weka/oe-adapt-default"
HF_TOKEN_SECRET_NAME="${HF_TOKEN_SECRET_NAME:-}"
REPO_GIT_REF=""
DRY_RUN=0

usage() {
    sed -n '2,35p' "$0"
    cat <<EOF

Options:
  --both                 launch one job per model: --vllm-model and --model-b
  --vllm-model MODEL     HF repo to serve (default: ${VLLM_MODEL})
  --model-b MODEL        second model, only used with --both (default: ${MODEL_B})
  --name NAME            served-model-name (default: lowercased model basename)
  --vllm-version VER     vLLM version for uvx (default: ${VLLM_VERSION})
  --cluster CLUSTER      beaker cluster (default: ${CLUSTER})
  --gpus N               GPUs per job (default: ${GPU_COUNT})
  --tp N                 tensor-parallel-size (default: ${TP_SIZE})
  --dp N                 data-parallel-size (default: gpus/tp)
  --max-model-len LEN    context window (default: ${MAX_MODEL_LEN})
  --gpu-util F           --gpu-memory-utilization (default: vllm default)
  --max-num-seqs N       --max-num-seqs (default: ${VLLM_MAX_NUM_SEQS})
  --vllm-extra-args S    free-form extra args for vllm serve
  --dataset DS           prompt source (default: ${DATASET})
  --num-prompts N        prompts to sample (default: ${NUM_PROMPTS})
  --num-samples N        completions per prompt (default: ${NUM_SAMPLES})
  --temperature F        default ${TEMPERATURE}
  --top-p F              default ${TOP_P}
  --max-tokens N         per-completion cap (default: ${MAX_TOKENS}); traces
                         that hit it are censored and reported as such
  --max-prompt-tokens N  skip longer prompts (default: ${MAX_PROMPT_TOKENS})
  --seed N               prompt-sampling seed (default: ${SEED}); MUST match
                         across the models being compared
  --concurrency N        in-flight requests (default: ${CONCURRENCY})
  --job-name NAME        beaker experiment name (default: think-len-<served-name>)
  --job-suffix S         appended to the default job name (e.g. -smoke)
  --priority PRI         beaker priority (default: ${PRIORITY})
  --preemptible          run preemptible (minRuntime 0, autoResume); default is
                         a protected, non-resumable job
  --task-timeout DUR     beaker job timeout (default: ${TASK_TIMEOUT})
  --budget BUDGET        default: ${BUDGET}
  --workspace WS         default: ${BEAKER_WORKSPACE}
  --image IMAGE          beaker image (default: ${BEAKER_IMAGE})
  --weka MOUNT           weka mount src:dst, or 'none' (default: ${WEKA_MOUNT})
  --hf-token-secret NAME beaker secret to expose as HF_TOKEN (default: none;
                         the default models and dataset are public)
  --repo-ref REF         git SHA/branch to run (default: current HEAD SHA)
  --dry-run              print the gantry command without submitting
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --both)              LAUNCH_BOTH=1; shift ;;
        --vllm-model)        VLLM_MODEL="$2"; shift 2 ;;
        --model-b)           MODEL_B="$2"; shift 2 ;;
        --name)              SERVED_MODEL_NAME="$2"; shift 2 ;;
        --vllm-version)      VLLM_VERSION="$2"; shift 2 ;;
        --cluster)           CLUSTER="$2"; shift 2 ;;
        --gpus)              GPU_COUNT="$2"; shift 2 ;;
        --tp)                TP_SIZE="$2"; shift 2 ;;
        --dp)                DP_SIZE="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util)          VLLM_GPU_UTIL="$2"; shift 2 ;;
        --max-num-seqs)      VLLM_MAX_NUM_SEQS="$2"; shift 2 ;;
        --vllm-extra-args)   VLLM_EXTRA_ARGS="$2"; shift 2 ;;
        --dataset)           DATASET="$2"; shift 2 ;;
        --num-prompts)       NUM_PROMPTS="$2"; shift 2 ;;
        --num-samples)       NUM_SAMPLES="$2"; shift 2 ;;
        --temperature)       TEMPERATURE="$2"; shift 2 ;;
        --top-p)             TOP_P="$2"; shift 2 ;;
        --max-tokens)        MAX_TOKENS="$2"; shift 2 ;;
        --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --concurrency)       CONCURRENCY="$2"; shift 2 ;;
        --job-name)          JOB_NAME="$2"; shift 2 ;;
        --job-suffix)        JOB_SUFFIX="$2"; shift 2 ;;
        --priority)          PRIORITY="$2"; shift 2 ;;
        --preemptible)       PREEMPTIBLE=1; shift ;;
        --task-timeout)      TASK_TIMEOUT="$2"; shift 2 ;;
        --budget)            BUDGET="$2"; shift 2 ;;
        --workspace)         BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)             BEAKER_IMAGE="$2"; shift 2 ;;
        --weka)              WEKA_MOUNT="$2"; shift 2 ;;
        --hf-token-secret)   HF_TOKEN_SECRET_NAME="$2"; shift 2 ;;
        --repo-ref)          REPO_GIT_REF="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help)           usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

if [ -z "$REPO_GIT_REF" ]; then
    REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo "warning: $REPO_GIT_REF doesn't appear on any remote branch." >&2
    echo "         the beaker job will fail to clone it. push first or pass --repo-ref." >&2
fi

submit_one() {
    local model="$1"
    local served="$2"

    if [ -z "$served" ]; then
        served="$(basename "$model" | tr '[:upper:]' '[:lower:]')"
    fi
    local dp="$DP_SIZE"
    if [ -z "$dp" ]; then dp=$(( GPU_COUNT / TP_SIZE )); fi
    local name="${JOB_NAME:-think-len-${served}}${JOB_SUFFIX}"

    cat <<EOF

=== Launching thinking-trace-length measurement on Beaker ===
  Model:      ${model}  (served as ${served}, vLLM ${VLLM_VERSION})
  Hardware:   ${CLUSTER}, ${GPU_COUNT} GPUs, TP=${TP_SIZE} DP=${dp}
  Context:    max_model_len=${MAX_MODEL_LEN}, max_tokens=${MAX_TOKENS}
  Prompts:    ${DATASET} -> ${NUM_PROMPTS} prompts x ${NUM_SAMPLES} samples (seed ${SEED})
  Sampling:   temperature=${TEMPERATURE} top_p=${TOP_P}
  Job:        ${name}  workspace=${BEAKER_WORKSPACE}  priority=${PRIORITY}
              preemptible=${PREEMPTIBLE}  task_timeout=${TASK_TIMEOUT}
  Repo ref:   ${REPO_GIT_REF}
EOF

    local cmd=(
        uvx --from beaker-gantry gantry run
        --yes
        --allow-dirty
        --timeout 0
        --workspace "$BEAKER_WORKSPACE"
        --name "$name"
        --description "Thinking-trace length on ${DATASET} with ${served} via vLLM on ${CLUSTER}"
        --ref "$REPO_GIT_REF"
        --cluster "$CLUSTER"
        --gpus "$GPU_COUNT"
        --priority "$PRIORITY"
        --task-timeout "$TASK_TIMEOUT"
        --beaker-image "$BEAKER_IMAGE"
        --budget "$BUDGET"
        --env "VLLM_MODEL=${model}"
        --env "SERVED_MODEL_NAME=${served}"
        --env "VLLM_VERSION=${VLLM_VERSION}"
        --env "VLLM_PORT=${VLLM_PORT}"
        --env "GPU_COUNT=${GPU_COUNT}"
        --env "TP_SIZE=${TP_SIZE}"
        --env "DP_SIZE=${dp}"
        --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
        --env "VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
        --env "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
        --env "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"
        --env "DATASET=${DATASET}"
        --env "NUM_PROMPTS=${NUM_PROMPTS}"
        --env "NUM_SAMPLES=${NUM_SAMPLES}"
        --env "TEMPERATURE=${TEMPERATURE}"
        --env "TOP_P=${TOP_P}"
        --env "MAX_TOKENS=${MAX_TOKENS}"
        --env "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}"
        --env "SEED=${SEED}"
        --env "CONCURRENCY=${CONCURRENCY}"
        --propagate-failure
        --no-python
    )
    if [ "$PREEMPTIBLE" = "1" ]; then
        cmd+=(--preemptible)
    else
        cmd+=(--not-preemptible)
    fi
    if [ "$WEKA_MOUNT" != "none" ]; then
        cmd+=(--weka "$WEKA_MOUNT")
    fi
    if [ -n "$HF_TOKEN_SECRET_NAME" ]; then
        cmd+=(--env-secret "HF_TOKEN=${HF_TOKEN_SECRET_NAME}")
    fi
    cmd+=(-- bash scripts/beaker/run_thinking_traces_in_job.sh)

    printf 'Launching with:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
    if [ "$DRY_RUN" = "1" ]; then
        echo "(dry run: not submitting)"
        return 0
    fi
    "${cmd[@]}"
}

submit_one "$VLLM_MODEL" "$SERVED_MODEL_NAME"
if [ "$LAUNCH_BOTH" = "1" ]; then
    # Deliberately reuses SEED/NUM_PROMPTS/MAX_PROMPT_TOKENS so both jobs
    # resolve to byte-identical prompts; analyze_traces.py verifies this from
    # the recorded prompt hashes rather than trusting it.
    submit_one "$MODEL_B" ""
fi
