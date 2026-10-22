#!/bin/bash
set -euo pipefail

cuda_version=12
build_only=false
while [[ $# -gt 0 ]]; do
case "$1" in
--build-only)
  build_only=true
  shift
  ;;
--cuda-version)
  if [[ $# -lt 2 ]]; then
    echo "Error: --cuda-version requires 12 or 13."
    exit 1
  fi
  cuda_version="${2:-}"
  shift 2
  ;;
--cuda-version=*)
  cuda_version="${1#*=}"
  shift
  ;;
*) break ;;
esac
done

if [[ "$cuda_version" != "12" && "$cuda_version" != "13" ]]; then
  echo "Error: --cuda-version must be 12 or 13."
  exit 1
fi

if [[ $# -eq 0 && "$build_only" == false ]]; then
  echo "Usage: $0 [--cuda-version 12|13] [--build-only] [SCRIPT [SCRIPT_ARGS...]]"
  exit 1
fi

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
image_name=open-instruct-integration-test-${sanitized_branch}-cuda${cuda_version}

beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

existing_image_desc=$(beaker image get "$beaker_user/$image_name" --format json 2>/dev/null | jq -r '.[0].description // ""' || echo "")

if [[ -n "$existing_image_desc" ]] && [[ "$existing_image_desc" == *"$git_hash"* ]]; then
  echo "Beaker image already exists for commit $git_hash, skipping Docker build and upload."
else
  echo "Creating CUDA $cuda_version beaker image for commit $git_hash..."
  CACHE_REPO="${DOCKER_CACHE_REPO:-ghcr.io/allenai/open-instruct:buildcache}"
  # BuildKit exposes this only to the dependency-install RUN, never as an image ENV/ARG.
  OI_BUILD_GITHUB_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [[ -z "$OI_BUILD_GITHUB_TOKEN" ]] && command -v gh >/dev/null 2>&1; then
    OI_BUILD_GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
  fi
  if [[ -z "$OI_BUILD_GITHUB_TOKEN" ]]; then
    echo "GitHub authentication is required to build the pinned private release plugins. Run gh auth login."
    exit 1
  fi
  export OI_BUILD_GITHUB_TOKEN

  # Try to build with cache push first, fall back to cache-from only if push fails
  if docker buildx build --platform=linux/amd64 \
    --secret id=github_token,env=OI_BUILD_GITHUB_TOKEN \
    --build-arg GIT_COMMIT="$git_hash" \
    --build-arg GIT_BRANCH="$git_branch" \
    --build-arg CUDA_VERSION="$cuda_version" \
    --cache-from "type=registry,ref=$CACHE_REPO" \
    --cache-to "type=registry,ref=$CACHE_REPO,mode=max" \
    --load \
    . -t "$image_name"; then
    echo "Build succeeded with cache push."
  else
    echo "Warning: Build with cache push failed (likely due to permissions). Retrying without cache push..."
    docker buildx build --platform=linux/amd64 \
      --secret id=github_token,env=OI_BUILD_GITHUB_TOKEN \
      --build-arg GIT_COMMIT="$git_hash" \
      --build-arg GIT_BRANCH="$git_branch" \
      --build-arg CUDA_VERSION="$cuda_version" \
      --cache-from "type=registry,ref=$CACHE_REPO" \
      --load \
      . -t "$image_name"
  fi

  beaker image rename "$beaker_user/$image_name" "" || echo "Image not found, skipping rename."

  beaker image create "$image_name" -n "$image_name" -w "ai2/$beaker_user" \
    --description "Git commit: $git_hash; CUDA: $cuda_version"
fi

if [[ "$build_only" == true ]]; then
  echo "Built image: $beaker_user/$image_name (commit $git_hash); no experiment submitted."
  exit 0
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
