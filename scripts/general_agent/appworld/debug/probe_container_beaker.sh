#!/bin/bash

# Cheap diagnostic (no training): start the AppWorld container on a Beaker podman host and
# report why it lives/dies + bridge-IP reachability. 1 node, 1 GPU (GPU unused; minimal for
# fast scheduling). See scripts/general_agent/appworld/debug/probe_container.py.
#
# Launch:
#   ./scripts/train/build_image_and_launch_dirty.sh scripts/general_agent/appworld/debug/probe_container_beaker.sh

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/jupiter \
       --image "$BEAKER_IMAGE" \
       --description "AppWorld container probe (podman start + reachability)" \
       --pure_docker_mode \
       --workspace ai2/general-tool-use \
       --priority urgent \
       --preemptible \
       --num_nodes 1 \
       --max_retries 1 \
       --env BEAKER_ALLOW_SUBCONTAINERS=1 \
       --env BEAKER_SKIP_DOCKER_SOCKET=1 \
       --env SWERL_PODMAN_SERVICE_COUNT=1 \
       --env MIRROR_URL=jupiter-cs-aus-193.reviz.ai2.in:5000 \
       --env PODMAN_NUM_LOCKS=65536 \
       --env CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf \
       --env DOCKERHUB_USERNAME=shashankg209 \
       --env APPWORLD_IMAGE=shatu/appworld-data:latest \
       --secret DOCKER_PAT=shashankg_DOCKER_PAT \
       --gpus 1 \
       --no_auto_dataset_cache \
       -- source scripts/docker/docker_login.sh \&\& python scripts/general_agent/appworld/debug/probe_container.py
