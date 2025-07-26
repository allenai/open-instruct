#!/bin/bash
set -e

image_name=open-instruct-dev

# Build and push the Docker image to Beaker
# Use --ssh default to forward SSH agent for git clone
echo "Building Docker image $image_name..."
DOCKER_BUILDKIT=1 docker build \
    --ssh default \
    -f Dockerfile.uv \
    --build-arg UV_CACHE_DIR=$UV_CACHE_DIR \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t $image_name \
    .

# Clean up Docker build cache to save space
docker builder prune -f || true

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image rename $beaker_user/$image_name "" || true

# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w ai2/$beaker_user

bash $1
