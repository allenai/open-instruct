#!/bin/bash
set -e

image_name=open-instruct-integration-test

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

# Ensure uv is installed and sync dependencies before running the script
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Python dependencies
echo "Installing dependencies with uv..."
uv sync --only-group dev

# Run the provided script
bash $1
