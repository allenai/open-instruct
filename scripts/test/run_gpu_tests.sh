#!/bin/bash
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

PYTEST_EXIT=0
if [[ $# -gt 0 ]]; then
    uv run pytest "$@" -xvs || PYTEST_EXIT=$?
else
    uv run pytest open_instruct/test_*_gpu.py -xvs || PYTEST_EXIT=$?
fi

cp -r open_instruct/test_data /output/test_data 2>/dev/null || true

exit $PYTEST_EXIT
