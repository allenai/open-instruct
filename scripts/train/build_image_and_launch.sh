#!/bin/bash
set -e

image_name=open-instruct-dev

# Build and push the Docker image to Beaker
echo "Building Docker image $image_name..."

# Use the regular Dockerfile which copies oe-eval-internal instead of cloning
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
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
