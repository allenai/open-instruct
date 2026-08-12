#!/bin/bash

# Score raw completions produced by generate_dolci_difficulty_completions.sh for
# per-prompt difficulty (16-sample pass-rate), completing the pipeline documented
# in research.md's Dolci difficulty-estimation entry.
#
# Downloads mnoukhov/dolci_think_rl_7b_messages_<MODEL_TAG> (raw completions) from
# the hub, converts to local jsonl, scores every sample with
# filter_existing_dataset_correctness.py using RLVR verifiers (math/ifeval are
# rule-based; code/code_stdio use a self-hosted code API brought up via
# code_api_setup.sh; general-quality/general-quality_ref use a standalone Qwen3-32B
# judge server, see configs/judge_configs/qwen3_32b_judge.yaml), and pushes the
# fully-annotated dataset (adds total_rollouts/total_correct_rollouts/passrate
# columns, no filtering: --lower_bound 0 --upper_bound 1 keeps every row) to
# mnoukhov/dolci_think_rl_7b_messages_<MODEL_TAG>_scored.
#
# Requests 8 GPUs (matching every other code_api_setup.sh caller in this repo) even
# though scoring itself is CPU-only, since code_api_setup.sh hardcodes 128 worker
# processes and needs a full node's CPU allocation to match.
#
# Requires a running LLM judge server - launch first if needed:
#   beaker experiment create configs/judge_configs/qwen3_32b_judge.yaml \
#       --name qwen3_32b_judge --workspace ai2/open-instruct-dev
# then resolve its hostname (beaker experiment get / beaker node get, see
# configs/judge_configs/qwen3_32b_judge.yaml) into a HOSTED_VLLM_API_BASE.
#
# Usage: [BEAKER_IMAGE=...] ./score_dolci_difficulty.sh MODEL_TAG HOSTED_VLLM_API_BASE
# (BEAKER_IMAGE is an env var override here, not a positional arg — with a variable
# number of following args there's no reliable way to tell an omitted image apart
# from MODEL_TAG, same reasoning as scripts/train/olmo-hybrid-small/gsm8k_grpo.sh.)
# e.g.:  ./score_dolci_difficulty.sh hybrid_450m http://saturn-cs-aus-240.reviz.ai2.in:8001/v1

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
SANITIZED_BRANCH=$(echo "$GIT_BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
BEAKER_IMAGE="${BEAKER_IMAGE:-${BEAKER_USER}/open-instruct-integration-test-${SANITIZED_BRANCH}-cuda12}"

MODEL_TAG="$1"
JUDGE_API_BASE="$2"
if [[ -z "$MODEL_TAG" || -z "$JUDGE_API_BASE" ]]; then
    echo "Usage: [BEAKER_IMAGE=...] $0 MODEL_TAG HOSTED_VLLM_API_BASE"
    exit 1
fi

DATASET="mnoukhov/dolci_think_rl_7b_messages_${MODEL_TAG}"
OUTPUT_DATASET="mnoukhov/dolci_think_rl_7b_messages_${MODEL_TAG}_scored"
EXP_NAME="dolci_difficulty_scoring_${MODEL_TAG}"
LOCAL_JSONL="/tmp/${MODEL_TAG}_completions.jsonl"
LOCAL_SCORED_JSONL="/tmp/${MODEL_TAG}_scored.jsonl"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Input dataset: $DATASET"
echo "Output dataset: $OUTPUT_DATASET"
echo "Judge API base: $JUDGE_API_BASE"

uv run python mason.py \
    --task_name "${EXP_NAME}" \
    --cluster ai2/saturn ai2/jupiter ai2/ceres \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-other \
    --env HOSTED_VLLM_API_BASE="$JUDGE_API_BASE" \
    --no_auto_dataset_cache \
    --gpus 8 \
    -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python scripts/data/rlvr/download_hub_dataset_as_jsonl.py \
        --dataset "$DATASET" \
        --split train \
        --output_file "$LOCAL_JSONL" \
    \&\& python scripts/data/rlvr/filter_existing_dataset_correctness.py \
        --files "$LOCAL_JSONL" \
        --annotate_original_dataset "$LOCAL_SCORED_JSONL" \
        --push_to_hub "$OUTPUT_DATASET" \
        --lower_bound 0.0 \
        --upper_bound 1.0 \
        --llm_judge_model hosted_vllm/Qwen/Qwen3-32B
