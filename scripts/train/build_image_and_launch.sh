#!/bin/bash
set -euo pipefail

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)
# Sanitize the branch name to remove invalid characters for Beaker names
# Beaker names can only contain letters, numbers, -_. and may not start with -
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | sed 's/^-//')
image_name=open-instruct-integration-test-${sanitized_branch}-${git_hash}

# Build the Docker image exactly like push-image.yml does, passing git info as build args
docker build --platform=linux/amd64 \
  --build-arg GIT_COMMIT="$git_hash" \
  --build-arg GIT_BRANCH="$git_branch" \
  . -t "$image_name"

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')


beaker image rename "$beaker_user/$image_name" "" || echo "Image not found, skipping rename."

# Create the image in the same workspace used for jobs
beaker image create "$image_name" -n "$image_name" -w "ai2/$beaker_user"

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
bash "$1" "$beaker_user/$image_name"
