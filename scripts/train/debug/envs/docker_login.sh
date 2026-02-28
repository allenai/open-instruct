#!/bin/bash
# Authenticate with Docker Hub using Beaker secret.
# Source this before training: source scripts/train/debug/envs/docker_login.sh
#
# Uses the Python docker SDK since the docker CLI isn't installed in the image.
if [ -n "$DOCKER_PAT" ]; then
    python -c "
import docker
client = docker.from_env()
client.login(username='hamishivi', password='$DOCKER_PAT')
print('Docker Hub login successful')
"
else
    echo "WARNING: DOCKER_PAT not set, skipping Docker Hub login (pulls will be rate-limited)"
fi
