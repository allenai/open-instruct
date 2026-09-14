#!/usr/bin/env bash
# Each arm starts cold; each rank retains its choices throughout the screen.
set -euo pipefail
export TRITON_CACHE_DIR="/tmp/trainer-capacity/rank-${RANK:?}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/trainer-capacity/rank-${RANK:?}/inductor"
exec python -m scripts.miles.profile_trainer_capacity "$@"
