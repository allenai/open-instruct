#!/bin/bash
set -euo pipefail

image_name=open-instruct-integration-test

# Build the Docker image exactly like push-image.yml does
docker build --platform linux/amd64 . -t $image_name

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')


beaker image rename $beaker_user/$image_name "" || echo "Image not found, skipping rename."

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
uv sync

# Run the provided script
bash $1 "$beaker_user/$image_name"
