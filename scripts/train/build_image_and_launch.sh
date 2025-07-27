#!/bin/bash
set -e

image_name=open-instruct-dev

# Build the Docker image exactly like push-image.yml does
docker build \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg CUDA=12.1.0 \
    --build-arg TARGET=cudnn8-devel \
    --build-arg DIST=ubuntu20.04 \
    --build-arg REQUIRE=requirements.txt \
    . \
    -t $image_name

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image rename $beaker_user/$image_name "" || true

# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w ai2/$beaker_user

bash $1
