#!/bin/bash
# Build (and optionally upload) the CUDA-13 open-instruct Beaker image.
#
# Differs from build_image_and_launch{,_dirty}.sh in three ways:
#   1. Passes --build-arg CUDA_VERSION=13 so the Dockerfile selects the cu130
#      base image + the `cuda13` uv dependency group.
#   2. Names the image with an explicit `-cu13` suffix so a CUDA-13 build is
#      never confused with (or overwrites) a cu12 build of the same branch.
#   3. Build-only by default; it does NOT launch any training. Pass --upload
#      to also create the Beaker image. It never pushes git.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

git_hash=$(git rev-parse --short HEAD)
git_branch=$(git rev-parse --abbrev-ref HEAD)
# Sanitize the branch name for Beaker (letters, numbers, -_. ; not leading -).
sanitized_branch=$(echo "$git_branch" | sed 's/[^a-zA-Z0-9._-]/-/g' | tr '[:upper:]' '[:lower:]' | sed 's/^-//')
image_name=open-instruct-integration-test-${sanitized_branch}-cu13

echo "Building $image_name (CUDA_VERSION=13, commit $git_hash)..."
docker build --platform=linux/amd64 \
  --build-arg CUDA_VERSION=13 \
  --build-arg GIT_COMMIT="$git_hash" \
  --build-arg GIT_BRANCH="$git_branch" \
  . -t "$image_name"
echo "Local image built: $image_name"

if [[ "${1:-}" == "--upload" ]]; then
  beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
  beaker image rename "$beaker_user/$image_name" "" || echo "No existing image to rename."
  beaker image create "$image_name" -n "$image_name" -w "ai2/oe-agents" \
    --description "Git commit: $git_hash (CUDA 13)"
  echo "Uploaded Beaker image: $beaker_user/$image_name"
else
  echo "Build-only (no upload). Re-run with --upload to create the Beaker image."
fi
