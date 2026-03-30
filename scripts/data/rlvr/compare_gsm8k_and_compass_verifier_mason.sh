#!/bin/bash

set -euo pipefail

EXP_NAME="${EXP_NAME:-compare_gsm8k_compass_verifier}"
CLUSTER="${CLUSTER:-ai2/saturn}"
WORKSPACE="${WORKSPACE:-ai2/oe-adapt-code}"
BEAKER_IMAGE="${BEAKER_IMAGE:-michaeln/open_instruct}"
PRIORITY="${PRIORITY:-normal}"
BUDGET="${BUDGET:-ai2/oe-adapt}"
DATASET="${DATASET:-mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets}"
SPLIT="${SPLIT:-test}"
JUDGE_MODEL="${JUDGE_MODEL:-opencompass/CompassVerifier-3B}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_LOCAL_DIR="${SAVE_LOCAL_DIR:-/weka/oe-adapt-default/michaeln/rlzero-open-instruct/results/gsm8k_compass_verifier}"

uv run python mason.py \
    --task_name "${EXP_NAME}" \
    --cluster "${CLUSTER}" \
    --workspace "${WORKSPACE}" \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --gpus 1 \
    --budget "${BUDGET}" \
    -- source configs/beaker_configs/ray_node_setup.sh \
\&\& UV_LINK_MODE=copy uv run --active python scripts/data/rlvr/compare_gsm8k_and_compass_verifier.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --judge-model "${JUDGE_MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --save-local-dir "${SAVE_LOCAL_DIR}" \
    "$@"
