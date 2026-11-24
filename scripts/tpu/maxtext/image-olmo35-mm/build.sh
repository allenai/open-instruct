#!/bin/bash
# Build the Olmo 3.5 multimodal SFT image with Cloud Build: MaxText at a pinned commit of jason718/maxtext, built with
# MaxText's own TPU dependencies Dockerfile (as olmo-gcp-infra's examples/olmo3-maxtext/cloudbuild.yaml builds
# us-docker.pkg.dev/ai2-olmo/olmo/maxtext:<commit>), then ./Dockerfile on top (CPU torch + OLMo-core for the olmo_mm
# loader). The tag is <maxtext commit>-olmocore-<first 9 of the OLMo-core commit>, the scheme of the reference run's
# image. As in ../image/build.sh, the Dockerfile travels inside the build config as a base64 tarball (no source
# upload), in steps of < 10,000 characters (Cloud Build's per-argument limit).
#   ./build.sh [run|e4|gates|small]   (default run; maxtext_source pins each)
#     run   217c8458: the reference run's image (jasonr-olmo35-mmsft-perf)
#     e4    ed83783e: the faster E4 flags (jasonr-olmo35-mmsft-perf2)
#     gates e3543ef2: perf2 + olmo_mm_batch_hash_steps, for ../k8s/olmo35-mm-sft-4x4x8-gate (jasonr-olmo35-mmsft-gates)
#     small e162d68a: gates + Small mm SFT config + prepacked text / image rows + keep_sources + producer loader, for
#           ../k8s/olmo35-small-mm-sft-* (jasonr-small-mm-sft); OLMo-core 7074cf40 (#942 head + #964, same loader as fd9bba1ba)
# An existing tag is never rebuilt or overwritten: if the registry already has it, the script says so and exits 0. Only
# the -olmocore- tag is pushed; the MaxText base it builds on stays local to the build.
# BUILD_SERVICE_ACCOUNT optionally names the account to build as (a full projects/<project>/serviceAccounts/<email>
# resource; the reference images were built as image-builder@ai2-olmo.iam.gserviceaccount.com).
set -euo pipefail
cd "$(dirname "$0")"
LABEL=${1:-run}
read -r _ REPO MAXTEXT_REF OLMO_CORE_REF < <(awk -v l="$LABEL" '$1 == l' maxtext_source) || true
if [[ -z "${MAXTEXT_REF:-}" ]]; then
  echo "unknown image '$LABEL'; maxtext_source has: $(cut -d' ' -f1 maxtext_source | tr '\n' ' ')" >&2
  exit 2
fi
IMAGE=us-docker.pkg.dev/ai2-olmo/olmo/maxtext
TAG="${MAXTEXT_REF}-olmocore-${OLMO_CORE_REF:0:9}"
if gcloud artifacts docker images describe "$IMAGE:$TAG" --project=ai2-olmo >/dev/null 2>&1; then
  echo "$IMAGE:$TAG exists already; not rebuilding it"
  echo "IMAGE=$IMAGE:$TAG"
  exit 0
fi
CTX=$(tar czf - Dockerfile | base64 | tr -d '\n')
cfg=$(mktemp "${TMPDIR:-/tmp}/maxtext-build.XXXXXX")
trap 'rm -f -- "$cfg"' EXIT
{
  cat <<YAML
steps:
- name: gcr.io/cloud-builders/git
  args: [clone, --no-checkout, "$REPO", maxtext]
- name: gcr.io/cloud-builders/git
  dir: maxtext
  args: [checkout, "$MAXTEXT_REF"]
- name: gcr.io/cloud-builders/docker
  dir: maxtext
  env: [DOCKER_BUILDKIT=1]
  args: [build, --file=src/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile, --build-arg=MODE=stable,
    --build-arg=WORKFLOW=pre-training, --build-arg=DEVICE=tpu, --build-arg=JAX_VERSION=NONE,
    --build-arg=LIBTPU_VERSION=NONE, --build-arg=PACKAGE_DIR=src, --build-arg=TF=false,
    "--tag=$IMAGE:$MAXTEXT_REF", .]
YAML
  for ((i = 0; i < ${#CTX}; i += 9000)); do
    echo "- name: bash"
    echo "  args: [\"-c\", \"printf %s ${CTX:i:9000} >> ctx.b64\"]"
  done
  cat <<YAML
- name: bash
  args: ["-c", "mkdir overlay && base64 -d ctx.b64 | tar xzf - -C overlay && rm ctx.b64"]
- name: gcr.io/cloud-builders/docker
  dir: overlay
  env: [DOCKER_BUILDKIT=1]
  args: [build, "--build-arg=BASE=$IMAGE:$MAXTEXT_REF", "--build-arg=OLMO_CORE_REF=$OLMO_CORE_REF", "--tag=$IMAGE:$TAG", .]
images: ["$IMAGE:$TAG"]
YAML
  if [[ -n "${BUILD_SERVICE_ACCOUNT:-}" ]]; then
    printf 'serviceAccount: %s\n' "$BUILD_SERVICE_ACCOUNT"
  fi
  cat <<YAML
options:
  machineType: E2_HIGHCPU_32
  logging: CLOUD_LOGGING_ONLY
timeout: 7200s
YAML
} > "$cfg"
echo "building $IMAGE:$TAG from $REPO@$MAXTEXT_REF + allenai/OLMo-core@$OLMO_CORE_REF"
gcloud builds submit --project=ai2-olmo --no-source --config="$cfg"
echo "IMAGE=$IMAGE:$TAG"
