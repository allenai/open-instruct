#!/bin/bash
set -euo pipefail

# 1) Verify we're inside a Git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: This directory is not a Git repository."
  exit 1
fi

# 2) Fail if there are any uncommitted or untracked changes
#    (staged, unstaged, or untracked files)
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "Error: Uncommitted changes detected. Please commit or stash before running."
  echo "------- git status (short) -------"
  git status --short
  exit 1
fi

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)
# Sanitize the branch name to remove invalid characters for Beaker names
# Beaker names can only contain letters, numbers, -_. and may not start with -
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
image_name=open-instruct-integration-test-${sanitized_branch}

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

existing_image_desc=$(beaker image get "$beaker_user/$image_name" --format json 2>/dev/null | jq -r '.[0].description // ""' || echo "")

if [[ -n "$existing_image_desc" ]] && [[ "$existing_image_desc" == *"$git_hash"* ]]; then
  echo "Beaker image already exists for commit $git_hash, skipping Docker build and upload."
else
  echo "Creating new beaker image for commit $git_hash..."
  CACHE_REPO="${DOCKER_CACHE_REPO:-ghcr.io/allenai/open-instruct:buildcache}"
  CACHE_ARGS="--cache-from type=registry,ref=$CACHE_REPO"
  [ "${DOCKER_CACHE_PUSH:-}" = "1" ] && CACHE_ARGS="$CACHE_ARGS --cache-to type=registry,ref=$CACHE_REPO,mode=max"

  docker buildx build --platform=linux/amd64 \
    --build-arg GIT_COMMIT="$git_hash" \
    --build-arg GIT_BRANCH="$git_branch" \
    $CACHE_ARGS \
    --load \
    . -t "$image_name"

  beaker image rename "$beaker_user/$image_name" "" || echo "Image not found, skipping rename."

  beaker image create "$image_name" -n "$image_name" -w "ai2/$beaker_user" --description "Git commit: $git_hash"
fi

# Ensure uv is installed and sync dependencies before running the script
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Python dependencies
echo "Installing dependencies with uv..."
uv sync

# Run the provided script with the image name and all remaining arguments
script="$1"
shift
bash "$script" "$beaker_user/$image_name" "$@"
