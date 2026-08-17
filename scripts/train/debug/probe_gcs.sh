#!/bin/bash
# Probe whether a Beaker job can read olmo-core checkpoints straight from GCS.
#
#   ./scripts/train/build_image_and_launch.sh --cuda-version 13 scripts/train/debug/probe_gcs.sh
#
# 0 GPUs, so it schedules in seconds. The answer decides whether the MoE spike
# can stream the ~320 GB s004 checkpoint from gs:// or has to stage it on WEKA
# first. olmo-core's io layer already speaks gs:// and google-cloud-storage is
# in the image, so the only open question is the credential.
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
    --secret GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS \
    --env GOOGLE_CLOUD_PROJECT=ai2-allennlp \
    -- uv run python scripts/train/debug/probe_gcs_access.py
