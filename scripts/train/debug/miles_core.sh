#!/bin/bash
set -euo pipefail
BEAKER_IMAGE="${1:?Pass the image from build_image_and_launch.sh --miles}"
# The pinned runtime is built against CUDA 13; Holmes carries a compatible driver.
# test_core_policy_contract imports an olmo-miles evaluation schema newer than the
# baked runtime, so it runs from its own launcher rather than this smoke.
uv run python mason.py \
    --cluster ai2/holmes \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Synthetic MILES/Core GPU update, serving, publication, and resume smoke test" \
    --pure_docker_mode \
    --num_nodes 1 \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    bash -euc 'cd /opt/core-rl
python -c "import torch; assert torch.cuda.is_available(), \"Pinned runtime requires a CUDA 13-compatible driver\"; print(torch.cuda.get_device_name(), torch.version.cuda)"
export PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH
python -m pytest tests/miles -q --ignore=tests/miles/test_core_policy_contract.py
python tests/miles/smoke.py /tmp/core-smoke
python tests/miles/smoke.py /tmp/core-smoke --resume'
