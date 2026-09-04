#!/usr/bin/env bash
#
# Launch the multi-model thinking-trace sweep as ONE Beaker job that serves each
# model in turn. Defaults target the frontier reasoning models on ai2/holmes
# (B300, 288 GB/GPU), sized so each fits at TP=4.
#
# Model slate (largest single-node-servable thinking model per family):
#   Qwen/Qwen3.5-397B-A17B-FP8    397B total /  17B active,  406 GB
#   moonshotai/Kimi-K2.6            1T total /  32B active,  595 GB
#   deepseek-ai/DeepSeek-V3.2-Exp 685B total /  37B active,  689 GB
#   zai-org/GLM-5.2-FP8           753B total / ~40B active,  756 GB
# Ordered cheapest first, so the least expensive model surfaces bugs earliest.
#
# Rejected for this slate, for the record: Qwen3.8-2.4T-A95B-FP8 needs 2496 GB
# and does not fit one node; DeepSeek-V4-* and Kimi-K3 ship no chat template;
# GLM-5.3-Flash and Qwen3.8-Flash-Next are not in the vLLM registry.
#
# Usage:
#   ./scripts/beaker/launch_thinking_traces_sweep.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODELS="Qwen/Qwen3.5-397B-A17B-FP8 moonshotai/Kimi-K2.6 deepseek-ai/DeepSeek-V3.2-Exp zai-org/GLM-5.2-FP8"
# 0.28 is the first release registering GlmMoeDsaForCausalLM, KimiK25ForConditionalGeneration,
# DeepseekV32ForCausalLM and Qwen3_5MoeForConditionalGeneration. B300 is sm_103 and needs the
# CUDA 13 image, so unlike the L40S runs this wants a NEW vLLM, not an old one.
VLLM_VERSION="0.28.0"
CLUSTER="ai2/holmes"
GPU_COUNT=4
TP_SIZE=""
MAX_MODEL_LEN=131072
MAX_TOKENS=128000
MAX_PROMPT_TOKENS=1536
DATASET="allenai/Dolci-Think-SFT-7B"
NUM_PROMPTS=1000
NUM_SAMPLES=8
TEMPERATURE=0.6
TOP_P=0.95
SEED=1234
CONCURRENCY=256
HF_REPO_ID=""
HF_TOKEN_SECRET_NAME="${HF_TOKEN_SECRET_NAME:-}"
JOB_NAME="think-len-sweep"
PRIORITY="normal"
MIN_RUNTIME="16h"
NO_AUTO_RESUME=0
TASK_TIMEOUT="72h"
BUDGET="ai2/oe-other"
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/olmo-instruct}"
BEAKER_IMAGE="${BEAKER_IMAGE:-ai2/cuda13.0-ubuntu22.04-torch2.11.0}"
WEKA_MOUNT="oe-adapt-default:/weka/oe-adapt-default"
REPO_GIT_REF=""
DRY_RUN=0

usage() {
    sed -n '2,20p' "$0"
    cat <<EOF

Options:
  --models "A B C"       HF repo ids, served in order (default: the slate above)
  --cluster CLUSTER      default: ${CLUSTER}
  --gpus N               GPUs / slots (default: ${GPU_COUNT})
  --tp N                 tensor-parallel size (default: --gpus)
  --num-prompts N        prompts per model (default: ${NUM_PROMPTS})
  --num-samples N        completions per prompt (default: ${NUM_SAMPLES})
  --max-model-len N      context (default: ${MAX_MODEL_LEN})
  --max-tokens N         generation cap (default: ${MAX_TOKENS})
  --concurrency N        in-flight requests (default: ${CONCURRENCY})
  --seed N               prompt-sampling seed (default: ${SEED})
  --hf-repo-id REPO      dataset repo to push traces to (default: none)
  --hf-token-secret NAME beaker secret exposed as HF_TOKEN
  --job-name NAME        default: ${JOB_NAME}
  --priority PRI         default: ${PRIORITY}
  --min-runtime DUR      guaranteed runtime before preemption (default: ${MIN_RUNTIME})
  --no-auto-resume       do not restart after preemption. Off by default here:
                         the sweep skips already-finished models on restart, so
                         auto-resume heals an unattended run instead of redoing work.
  --task-timeout DUR     default: ${TASK_TIMEOUT}
  --workspace WS         default: ${BEAKER_WORKSPACE}
  --image IMAGE          default: ${BEAKER_IMAGE}
  --repo-ref REF         git SHA/branch (default: current HEAD)
  --dry-run              print the gantry command without submitting
EOF
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models)            MODELS="$2"; shift 2 ;;
        --cluster)           CLUSTER="$2"; shift 2 ;;
        --gpus)              GPU_COUNT="$2"; shift 2 ;;
        --tp)                TP_SIZE="$2"; shift 2 ;;
        --num-prompts)       NUM_PROMPTS="$2"; shift 2 ;;
        --num-samples)       NUM_SAMPLES="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-tokens)        MAX_TOKENS="$2"; shift 2 ;;
        --concurrency)       CONCURRENCY="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --hf-repo-id)        HF_REPO_ID="$2"; shift 2 ;;
        --hf-token-secret)   HF_TOKEN_SECRET_NAME="$2"; shift 2 ;;
        --job-name)          JOB_NAME="$2"; shift 2 ;;
        --priority)          PRIORITY="$2"; shift 2 ;;
        --min-runtime)       MIN_RUNTIME="$2"; shift 2 ;;
        --no-auto-resume)    NO_AUTO_RESUME=1; shift ;;
        --task-timeout)      TASK_TIMEOUT="$2"; shift 2 ;;
        --workspace)         BEAKER_WORKSPACE="$2"; shift 2 ;;
        --image)             BEAKER_IMAGE="$2"; shift 2 ;;
        --weka)              WEKA_MOUNT="$2"; shift 2 ;;
        --repo-ref)          REPO_GIT_REF="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help)           usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

[ -n "$TP_SIZE" ] || TP_SIZE="$GPU_COUNT"
[ -n "$REPO_GIT_REF" ] || REPO_GIT_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if ! git -C "$REPO_ROOT" branch -r --contains "$REPO_GIT_REF" 2>/dev/null | grep -q .; then
    echo "warning: $REPO_GIT_REF is not on any remote branch; the job will fail to clone it." >&2
fi

cat <<EOF

=== Thinking-trace sweep ===
  Models      : ${MODELS}
  Hardware    : ${CLUSTER}, ${GPU_COUNT} GPUs (TP=${TP_SIZE}), vLLM ${VLLM_VERSION}
  Context     : max_model_len=${MAX_MODEL_LEN}  max_tokens=${MAX_TOKENS}
  Sampling    : ${NUM_PROMPTS} prompts x ${NUM_SAMPLES} samples (seed ${SEED}), concurrency ${CONCURRENCY}
  Hub         : ${HF_REPO_ID:-<none>}  secret=${HF_TOKEN_SECRET_NAME:-<none>}
  Job         : ${JOB_NAME}  ws=${BEAKER_WORKSPACE}  priority=${PRIORITY}
                min_runtime=${MIN_RUNTIME}  auto_resume=$([ "$NO_AUTO_RESUME" = 1 ] && echo off || echo on)  timeout=${TASK_TIMEOUT}
  Ref         : ${REPO_GIT_REF}
EOF

cmd=(
    uvx --from beaker-gantry gantry run --yes --allow-dirty --timeout 0
    --workspace "$BEAKER_WORKSPACE" --name "$JOB_NAME"
    --description "Thinking-trace length sweep over 4 frontier reasoning models on ${DATASET}"
    --ref "$REPO_GIT_REF" --cluster "$CLUSTER" --gpus "$GPU_COUNT"
    --priority "$PRIORITY" --min-runtime "$MIN_RUNTIME" --task-timeout "$TASK_TIMEOUT"
    --beaker-image "$BEAKER_IMAGE" --budget "$BUDGET"
    --env "MODELS=${MODELS}" --env "VLLM_VERSION=${VLLM_VERSION}"
    --env "GPU_COUNT=${GPU_COUNT}" --env "TP_SIZE=${TP_SIZE}"
    --env "MAX_MODEL_LEN=${MAX_MODEL_LEN}" --env "MAX_TOKENS=${MAX_TOKENS}"
    --env "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS}" --env "DATASET=${DATASET}"
    --env "NUM_PROMPTS=${NUM_PROMPTS}" --env "NUM_SAMPLES=${NUM_SAMPLES}"
    --env "TEMPERATURE=${TEMPERATURE}" --env "TOP_P=${TOP_P}" --env "SEED=${SEED}"
    --env "CONCURRENCY=${CONCURRENCY}" --env "HF_REPO_ID=${HF_REPO_ID}"
    --propagate-failure --no-python
)
[ "$NO_AUTO_RESUME" = "1" ] && cmd+=(--no-auto-resume)
[ "$WEKA_MOUNT" != "none" ] && cmd+=(--weka "$WEKA_MOUNT")
[ -n "$HF_TOKEN_SECRET_NAME" ] && cmd+=(--env-secret "HF_TOKEN=${HF_TOKEN_SECRET_NAME}")
cmd+=(-- bash scripts/beaker/run_thinking_traces_sweep_in_job.sh)

printf 'Launching with:'; printf ' %q' "${cmd[@]}"; printf '\n\n'
if [ "$DRY_RUN" = "1" ]; then echo "(dry run: not submitting)"; exit 0; fi
"${cmd[@]}"
