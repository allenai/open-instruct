#!/bin/bash
# Smoke-test the Apptainer backend for SWERL Sandbox on Tillicum (or any Slurm
# cluster with apptainer on PATH).
#
# This does NOT launch training. It just runs the eight-step backend probe in
# scripts/debug/apptainer_backend_smoke.py end-to-end:
#   - instance start + writable tmpfs + fakeroot
#   - run_command / write_file / read_file / put_archive
#   - state persistence across exec calls
#   - timeout wrapper
#
# Usage (from a login node):
#   salloc --qos=debug --gpus=1 --cpus-per-task=8 --mem=100G --time=00:15:00
#   bash scripts/train/debug/envs/swerl_sandbox_apptainer_smoke.sh
#
# Or as a single command, dropping you straight onto a compute node:
#   salloc --qos=debug --gpus=1 --cpus-per-task=8 --mem=100G --time=00:15:00 \
#     bash scripts/train/debug/envs/swerl_sandbox_apptainer_smoke.sh
#
# Override the image by setting $APPTAINER_TEST_IMAGE, e.g.
#   APPTAINER_TEST_IMAGE=/shared/sif/ubuntu2204.sif bash scripts/train/debug/envs/swerl_sandbox_apptainer_smoke.sh

set -euo pipefail

IMAGE="${APPTAINER_TEST_IMAGE:-docker://ubuntu:22.04}"
CACHE_DIR="${APPTAINER_CACHEDIR:-/tmp/${USER:-apptainer}-apptainer-cache}"
TMP_DIR="${APPTAINER_TMPDIR:-/tmp/${USER:-apptainer}-apptainer-tmp}"

mkdir -p "$CACHE_DIR" "$TMP_DIR"

echo "Host:      $(hostname)"
echo "Image:     $IMAGE"
echo "Cache:     $CACHE_DIR"
echo "Tmp:       $TMP_DIR"
apptainer --version
echo

uv run python scripts/debug/apptainer_backend_smoke.py \
    --image "$IMAGE" \
    --cache-dir "$CACHE_DIR" \
    --tmp-dir "$TMP_DIR" \
    --timeout 5 \
    "$@"
