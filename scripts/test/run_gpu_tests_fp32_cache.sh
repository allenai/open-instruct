#!/bin/bash
# GPU tests with fp32 LM head CACHE mode (runs inside container)
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

export OPEN_INSTRUCT_FP32_LM_HEAD=1
echo "Running GPU tests with FP32 LM head CACHE mode (OPEN_INSTRUCT_FP32_LM_HEAD=1)"

PYTEST_EXIT=0
uv run pytest open_instruct/test_*_gpu.py -xvs || PYTEST_EXIT=$?

cp -r open_instruct/test_data /output/test_data 2>/dev/null || true

exit $PYTEST_EXIT
