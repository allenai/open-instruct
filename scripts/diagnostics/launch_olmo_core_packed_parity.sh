#!/bin/bash
# One-shot Beaker launch for the OLMo-core packed intra-doc attention parity probe.
# Usage: ./scripts/train/build_image_and_launch.sh scripts/diagnostics/launch_olmo_core_packed_parity.sh

EXP_NAME="${EXP_NAME:-olmo_core_packed_parity}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
shift || true

PRIORITY="${PRIORITY:-urgent}"

uv run mason.py \
    --task_name "${EXP_NAME}" \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/open-instruct-dev \
    --priority "${PRIORITY}" \
    --pure_docker_mode \
    --image "${BEAKER_IMAGE}" \
    --preemptible \
    --num_nodes 1 \
    --gpus 1 \
    --budget ai2/oe-adapt \
    --artifact_ttl 1d \
    --no_auto_dataset_cache \
    -- \
    uv run python scripts/diagnostics/olmo_core_packed_parity.py "$@"
