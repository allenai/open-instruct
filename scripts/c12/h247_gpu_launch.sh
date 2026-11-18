#!/usr/bin/env bash
set -euo pipefail

image="${1:?CUDA 13 Beaker image}"
shift
IMAGE="$image" bash "$(dirname "$0")/h161_gpu_launch.sh" "$@"
