#!/usr/bin/env bash
# Keep each rank's autotuning choices across fresh-process checkpoint resumes.
set -euo pipefail
export TRITON_CACHE_DIR="${RUN_ROOT:?}/triton-cache/rank-${RANK:?}"
exec python -m scripts.miles.checkpoint_benchmark "$@"
