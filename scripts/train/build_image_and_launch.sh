#!/bin/bash
set -euo pipefail

image_name=open-instruct-dev

# Build and push the Docker image to Beaker
docker build -f Dockerfile.uv --build-arg UV_CACHE_DIR="$UV_CACHE_DIR" -t $image_name .

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image rename "$beaker_user/$image_name" "" || true

# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w "ai2/$beaker_user"

bash "$1" "$beaker_user/$image_name"
