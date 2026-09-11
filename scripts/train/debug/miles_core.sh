#!/bin/bash
set -euo pipefail
BEAKER_IMAGE="${1:?Pass the image from build_image_and_launch.sh --miles}"
# The pinned runtime is built against CUDA 13; Holmes carries a compatible driver.
# mason.py joins the trailing tokens with spaces into one /bin/bash -c string, so
# the steps are chained with && rather than wrapped in a quoted bash -e body,
# which would lose its quoting and stop gating on failures.
# Ignored modules need sources newer than the baked runtime (olmo-miles evaluation
# schemas, researcher run files, replay/update-zero launchers) and have their own
# launchers; they are not part of the Core update/serving/resume smoke.
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
    cd /opt/core-rl '&&' export 'PYTHONPATH=/opt/core-rl/tests/miles:$PYTHONPATH' '&&' \
    python -m pytest tests/miles -q \
        --ignore=tests/miles/test_core_policy_contract.py \
        --ignore=tests/miles/test_control_exercise.py \
        --ignore=tests/miles/test_light_sft_gsm8k.py \
        --ignore=tests/miles/test_replay_diagnostics.py \
        --ignore=tests/miles/test_update_zero_gradient_capture.py \
        --ignore=tests/miles/test_update_zero_megatron.py \
    '&&' python tests/miles/smoke.py /tmp/core-smoke \
    '&&' python tests/miles/smoke.py /tmp/core-smoke --resume
