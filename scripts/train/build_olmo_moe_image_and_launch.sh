#!/bin/bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: This directory is not a Git repository."
  exit 1
fi

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "Error: Uncommitted changes detected. Please commit or stash before running."
  git status --short
  exit 1
fi

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
base_image_name=open-instruct-integration-test-${sanitized_branch}-olmo-moe-base
image_name=open-instruct-integration-test-${sanitized_branch}-olmo-moe
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

existing_image_desc=$(beaker image get "$beaker_user/$image_name" --format json 2>/dev/null | jq -r '.[0].description // ""' || echo "")

build_docker_image() {
  local tag=$1
  local dockerfile=$2
  local cache_repo=$3
  shift 3
  local extra_args=("$@")

  if docker buildx build --platform=linux/amd64 \
    --file "$dockerfile" \
    --build-arg GIT_COMMIT="$git_hash" \
    --build-arg GIT_BRANCH="$git_branch" \
    --cache-from "type=registry,ref=$cache_repo" \
    --cache-to "type=registry,ref=$cache_repo,mode=max" \
    --load \
    "${extra_args[@]}" \
    --tag "$tag" \
    .; then
    echo "Build succeeded with cache push."
  else
    echo "Warning: Build with cache push failed. Retrying without cache push..."
    docker buildx build --platform=linux/amd64 \
      --file "$dockerfile" \
      --build-arg GIT_COMMIT="$git_hash" \
      --build-arg GIT_BRANCH="$git_branch" \
      --cache-from "type=registry,ref=$cache_repo" \
      --load \
      "${extra_args[@]}" \
      --tag "$tag" \
      .
  fi
}

if [[ -n "$existing_image_desc" ]] && [[ "$existing_image_desc" == *"$git_hash"* ]]; then
  echo "Beaker OLMo MoE image already exists for commit $git_hash, skipping build and upload."
else
  cache_repo="${DOCKER_CACHE_REPO:-ghcr.io/allenai/open-instruct:buildcache}"
  build_docker_image "$base_image_name" Dockerfile "$cache_repo"
  build_docker_image \
    "$image_name" \
    Dockerfile.olmo-moe \
    "${cache_repo}-olmo-moe" \
    --build-arg "OPEN_INSTRUCT_BASE_IMAGE=$base_image_name"

  beaker image rename "$beaker_user/$image_name" "" || echo "Image not found, skipping rename."
  beaker image create \
    "$image_name" \
    --name "$image_name" \
    --workspace "ai2/$beaker_user" \
    --description "Git commit: $git_hash; variant: olmo-moe"
fi

if [[ "${OPEN_INSTRUCT_BUILD_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Installing dependencies with uv..."
uv sync

script="$1"
shift
bash "$script" "$beaker_user/$image_name" "$@"
