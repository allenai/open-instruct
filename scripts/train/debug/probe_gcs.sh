#!/bin/bash
# Probe whether a Beaker job can read olmo-core checkpoints straight from GCS.
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 scripts/train/debug/probe_gcs.sh
#
# 0 GPUs, so it schedules in seconds.
#
# VERIFIED WORKING 2026-08-18 (01M0BA1DE6H77WXCY8ZG6DY1XW): olmo-core's io layer
# speaks gs:// natively and google-cloud-storage is in the image, so checkpoints
# can be read straight from the bucket with no WEKA staging step.
#
# The credential was the whole problem. Notes for whoever hits this next:
#
# * The workspace-level GOOGLE_APPLICATION_CREDENTIALS secret in
#   ai2/open-instruct-dev is a dead personal ADC token (authorized_user, 2025-09)
#   that fails with `invalid_grant`, and only its author can overwrite it. Use a
#   user-scoped secret instead -- mason.py already prefers "<user>_<NAME>" for
#   the secrets in its `useful_secrets` list.
# * The secret's value is a PATH to a service-account key on Weka, not the key
#   itself. Write it with the value as an argument, not via `<<<`: a trailing
#   newline makes the path unresolvable, and google.auth does not strip it.
# * On-prem clusters (jupiter/ceres/holmes) have no GCE metadata server, so there
#   is no ambient node service account to fall back on. Augusta does.
set -euo pipefail

BEAKER_IMAGE="${1:-abhishekr/open-instruct-integration-test-spike-s004-moe-cuda13}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Probe direct GCS read of the s004 checkpoint from a Beaker job" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 0 \
    --non_resumable \
    --no_auto_dataset_cache \
    --secret GOOGLE_APPLICATION_CREDENTIALS="${GCS_SECRET:-abhishekr_GOOGLE_APPLICATION_CREDENTIALS}" \
    -- uv run python scripts/train/debug/probe_gcs_access.py
