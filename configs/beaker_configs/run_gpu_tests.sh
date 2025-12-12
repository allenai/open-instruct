#!/bin/bash

source configs/beaker_configs/ray_node_setup.sh

uv run pytest open_instruct/test_*_gpu.py -xvs
PYTEST_EXIT=$?

cp -r open_instruct/test_data /output/test_data 2>/dev/null || true

exit $PYTEST_EXIT
