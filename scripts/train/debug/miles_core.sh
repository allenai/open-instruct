#!/bin/bash
set -euo pipefail
BEAKER_IMAGE="${1:?Pass the image from build_image_and_launch.sh --miles}"
uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority normal \
    --image "$BEAKER_IMAGE" \
    --description "Synthetic MILES/Core GPU update, serving, publication, and resume smoke test" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    bash -euc 'cd /opt/core-rl
export PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH
python -m pytest tests/miles -q
python tests/miles/smoke.py /tmp/core-smoke
python tests/miles/smoke.py /tmp/core-smoke --resume'
