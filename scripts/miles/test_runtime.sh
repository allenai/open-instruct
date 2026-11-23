#!/bin/bash
# Run inside an application image built with runtime/miles/Dockerfile.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -m pytest --require-miles-runtime tests/miles "$@"
