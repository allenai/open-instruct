#!/bin/bash
# Build the Olmo post-training MaxText image with Cloud Build. The image-builder service account cannot read
# uploaded source, so the Dockerfile and patches/ travel inside the build config as a base64 tarball, split into
# steps of < 10,000 characters (Cloud Build's per-argument limit) that append to one file in the shared workspace.
#   ./build.sh   (H072: r2's image + the batch-hash logging patch)
set -euo pipefail
cd "$(dirname "$0")"
../fetch_patches.sh
TAG="51f8cc4b-p907d86f3-rf$(cat Dockerfile patches/* | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:8])')"
IMAGE=us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain
CTX=$(tar czf - Dockerfile patches | base64 | tr -d '\n')
cfg=$(mktemp "${TMPDIR:-/tmp}/maxtext-build.XXXXXX")
trap 'rm -f -- "$cfg"' EXIT
{
  echo "steps:"
  for ((i = 0; i < ${#CTX}; i += 9000)); do
    echo "- name: bash"
    echo "  args: [\"-c\", \"printf %s ${CTX:i:9000} >> ctx.b64\"]"
  done
  cat <<YAML
- name: bash
  args: ["-c", "base64 -d ctx.b64 | tar xzf - && rm ctx.b64"]
- name: gcr.io/cloud-builders/docker
  args: ["build", "--tag=$IMAGE:$TAG", "."]
images: ["$IMAGE:$TAG"]
YAML
  if [[ -n "${BUILD_SERVICE_ACCOUNT:-}" ]]; then
    printf 'serviceAccount: %s\n' "$BUILD_SERVICE_ACCOUNT"
  fi
  cat <<YAML
options:
  machineType: E2_HIGHCPU_8
  logging: CLOUD_LOGGING_ONLY
timeout: 3600s
YAML
} > "$cfg"
echo "building $IMAGE:$TAG ($(grep -c 'ctx.b64"' "$cfg") context chunks)"
gcloud builds submit --project=ai2-olmo --no-source --config="$cfg"
echo "IMAGE=$IMAGE:$TAG"
