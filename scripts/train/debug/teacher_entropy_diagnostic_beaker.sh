#!/bin/bash

# Teacher-entropy diagnostic (OPD validation program step 5) on one GPU:
# student rollouts via vLLM, then exact teacher entropy / top-k statistics per token
# via scripts/eopd/teacher_entropy_diagnostic.py. Defaults reproduce the EOPD setting
# (Qwen3-4B-Base student, Qwen3-8B non-thinking teacher, DAPO-Math-14k prompts).
#
#   STUDENT=... TEACHER=... PROMPTS=... LABEL=... \
#     CODE_REF=codex/qwen35-math-opd ./scripts/train/debug/teacher_entropy_diagnostic_beaker.sh IMAGE
#
# CODE_REF (optional) clones that pushed ref inside the job and copies its open_instruct/
# package over the image's editable install, like grpo_fast_opd_trace_audit_beaker.sh.

set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
STUDENT="${STUDENT:-Qwen/Qwen3-4B-Base}"
# Set a revision to the empty string to omit it (e.g. for a local checkpoint directory).
STUDENT_REVISION="${STUDENT_REVISION-906bfd4b4dc7f14ee4320094d8b41684abff8539}"
TEACHER="${TEACHER:-Qwen/Qwen3-8B}"
TEACHER_REVISION="${TEACHER_REVISION-b968826d9c46dd6066d109eabc6255188de91218}"
PROMPTS="${PROMPTS:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/miles-opd/data/eopd-math-v1/dapo_math_14k.jsonl}"
LABEL="${LABEL:-qwen3-8b_on_qwen3-4b-base_dapo14k}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUTPUT="${OUTPUT:-/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/eopd/teacher_entropy/$(date -u +%Y%m%dT%H%M%SZ)_${LABEL}}"
CODE_REF="${CODE_REF:-}"
SCRIPT="scripts/eopd/teacher_entropy_diagnostic.py"
SETUP="true"
if [[ -n "$CODE_REF" ]]; then
    ORIGIN_URL=$(git remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#')
    SETUP="git clone --depth 1 --branch $CODE_REF $ORIGIN_URL /tmp/oi && cp -r /tmp/oi/open_instruct/. /stage/open_instruct/"
    SCRIPT="/tmp/oi/$SCRIPT"
fi

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Output: $OUTPUT"
[[ -n "$CODE_REF" ]] && echo "Running code from $CODE_REF"

uv run python mason.py \
       --cluster ai2/jupiter \
       --cluster ai2/saturn \
       --image "$BEAKER_IMAGE" \
       --description "Teacher-entropy diagnostic: $LABEL" \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --env VLLM_DISABLE_COMPILE_CACHE=1 \
       --env VLLM_USE_V1=1 \
       --budget ai2/oe-other \
       --gpus 1 \
       --no_auto_dataset_cache \
       -- $SETUP \&\& python "$SCRIPT" all \
    --student "$STUDENT" ${STUDENT_REVISION:+--student-revision "$STUDENT_REVISION"} \
    --teacher "$TEACHER" ${TEACHER_REVISION:+--teacher-revision "$TEACHER_REVISION"} \
    --prompts "$PROMPTS" --num-prompts "$NUM_PROMPTS" --max-tokens "$MAX_TOKENS" \
    --temperature 1.0 --top-p 1.0 --k 16 --tau 0.8 \
    --output "$OUTPUT"
