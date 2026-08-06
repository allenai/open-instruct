#!/bin/bash

# Generate 16 completions per prompt for mnoukhov/dolci_think_rl_7b_messages
# using an OLMo-hybrid checkpoint, as an input to difficulty estimation
# (pass-rate over K samples). Shards the dataset across 8 GPUs on a single
# node (see generate_dolci_difficulty_shards_worker.sh), then combines and
# pushes the raw completions to mnoukhov/dolci_think_rl_7b_messages_<MODEL_TAG>.
#
# Scoring (pass-rate via RLVR verifiers / LLM judge) is a separate step, run
# with scripts/data/rlvr/filter_existing_dataset_correctness.py afterward.
#
# Usage: ./generate_dolci_difficulty_completions.sh [BEAKER_IMAGE] MODEL_PATH MODEL_TAG
# e.g.:  ./generate_dolci_difficulty_completions.sh \
#          /weka/oe-training-default/ai2-llm/checkpoints/saumyam/hybrid-small-sft-think-450m-lr8e-4/step24834-hf \
#          hybrid_450m

DATASET="mnoukhov/dolci_think_rl_7b_messages"
DATASET_SPLIT="train"
DATASET_SIZE=102014
NUM_GPUS=8
NUM_SAMPLES=16
TEMPERATURE=1.0

# Default to this branch's cuda12 integration test image, as built by
# scripts/train/build_image_and_launch.sh --cuda-version 12.
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
SANITIZED_BRANCH=$(echo "$GIT_BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test-${SANITIZED_BRANCH}-cuda12}"
[[ $# -gt 0 ]] && shift

MODEL_PATH="$1"
MODEL_TAG="$2"
if [[ -z "$MODEL_PATH" || -z "$MODEL_TAG" ]]; then
    echo "Usage: $0 [BEAKER_IMAGE] MODEL_PATH MODEL_TAG"
    exit 1
fi

EXP_NAME="dolci_difficulty_completions_${MODEL_TAG}"

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Model: $MODEL_PATH"
echo "Output dataset: mnoukhov/dolci_think_rl_7b_messages_${MODEL_TAG}"

uv run python mason.py \
    --task_name "${EXP_NAME}" \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --pure_docker_mode \
    --image "$BEAKER_IMAGE" \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-other \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --artifact_ttl 1d \
    --gpus "$NUM_GPUS" \
    -- bash scripts/data/rlvr/generate_dolci_difficulty_shards_worker.sh \
        "$MODEL_PATH" "$MODEL_TAG" "$DATASET" "$DATASET_SPLIT" "$DATASET_SIZE" "$NUM_GPUS" "$NUM_SAMPLES" "$TEMPERATURE"
