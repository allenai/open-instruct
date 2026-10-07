#!/bin/bash
# Build the Olmo post-training MaxText image with Cloud Build. The image-builder service account cannot read
# uploaded source, so the Dockerfile and patches/ travel inside the build config as a base64 tarball.
#   ./build.sh <maxtext base commit>
set -euo pipefail
cd "$(dirname "$0")"
BASE_REF=${1:?maxtext base commit}
TAG="${BASE_REF:0:8}-p$(cat Dockerfile patches/* | sha256sum | cut -c1-8)"
IMAGE=us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain
CTX=$(tar czf - Dockerfile patches | base64 -w0)
cfg=$(mktemp --suffix=.yaml)
cat > "$cfg" <<YAML
steps:
- name: bash
  args: ["-c", "echo $CTX | base64 -d | tar xzf -"]
- name: gcr.io/cloud-builders/docker
  args: ["build", "--build-arg=BASE_REF=$BASE_REF", "--tag=$IMAGE:$TAG", "."]
images: ["$IMAGE:$TAG"]
serviceAccount: projects/ai2-olmo/serviceAccounts/image-builder@ai2-olmo.iam.gserviceaccount.com
options:
  machineType: E2_HIGHCPU_8
  logging: CLOUD_LOGGING_ONLY
timeout: 3600s
YAML
echo "building $IMAGE:$TAG"
gcloud builds submit --project=ai2-olmo --no-source --config="$cfg"
echo "IMAGE=$IMAGE:$TAG"
