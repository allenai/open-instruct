#!/bin/bash
# Authenticate with Docker Hub using Beaker secret.
# Source this before training: source scripts/train/debug/envs/docker_login.sh
if [ -n "$DOCKER_PAT" ]; then
    echo "$DOCKER_PAT" | docker login -u hamishivi --password-stdin
else
    echo "WARNING: DOCKER_PAT not set, skipping Docker Hub login (pulls will be rate-limited)"
fi
