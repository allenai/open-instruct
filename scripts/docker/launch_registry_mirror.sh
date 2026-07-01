#!/usr/bin/env bash
set -euo pipefail

# Launch a single Docker Hub pull-through registry cache on Beaker.
#
# Required env:
#   DOCKER_HUB_USERNAME        Docker Hub username for the upstream proxy.
#   PERSONAL_ACCESS_TOKEN      Docker Hub PAT. DOCKER_PAT is also accepted.
#
# Optional env:
#   WORKSPACE                  Beaker workspace. Defaults to ai2/open-instruct-dev.
#   BUDGET                     Beaker budget. Defaults to ai2/compute.
#   CLUSTER                    Beaker cluster. Defaults to ai2/jupiter.
#   SECRET_NAME                Beaker secret name. Defaults to registry-secret.
#   EXPERIMENT_NAME            Name to assign after launch. Defaults to registry-mirror.
#   REGISTRY_PORT              Host-networked registry port. Defaults to 5000.
#   PRIORITY                   Beaker priority. Defaults to normal.
#   MIN_RUNTIME                Beaker minRuntime. Defaults to 336h0m0s (2 weeks).
#
# Usage:
#   DOCKER_HUB_USERNAME=... PERSONAL_ACCESS_TOKEN=... \
#     scripts/docker/launch_registry_mirror.sh

WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}"
BUDGET="${BUDGET:-ai2/oe-adapt}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
SECRET_NAME="${SECRET_NAME:-registry-secret}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-registry-mirror}"
REGISTRY_PORT="${REGISTRY_PORT:-5000}"
PRIORITY="${PRIORITY:-urgent}"
MIN_RUNTIME="${MIN_RUNTIME:-8h0m0s}"

DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:?Set DOCKER_HUB_USERNAME}"
PERSONAL_ACCESS_TOKEN="${PERSONAL_ACCESS_TOKEN:-${DOCKER_PAT:-}}"
if [[ -z "${PERSONAL_ACCESS_TOKEN}" ]]; then
    echo "Set PERSONAL_ACCESS_TOKEN or DOCKER_PAT" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

config_file="${tmp_dir}/config.yml"
beaker_file="${tmp_dir}/beaker.yml"

cat >"${config_file}" <<EOF
version: 0.1
log:
  fields:
    service: registry
storage:
  filesystem:
    rootdirectory: /var/lib/registry
http:
  addr: :${REGISTRY_PORT}
proxy:
  remoteurl: https://registry-1.docker.io
  username: ${DOCKER_HUB_USERNAME}
  password: ${PERSONAL_ACCESS_TOKEN}
EOF

echo "Writing Beaker secret ${SECRET_NAME} in workspace ${WORKSPACE}"
beaker secret write --workspace "${WORKSPACE}" "${SECRET_NAME}" "$(cat "${config_file}")"

cat >"${beaker_file}" <<EOF
version: v2
description: Docker Hub Local Pull-Through Cache Registry
budget: ${BUDGET}
tasks:
  - name: registry-mirror
    image:
      docker: registry:3
    command: [/bin/sh, -c, registry serve /etc/docker/registry/config.yml]
    datasets:
      - mountPath: /etc/docker/registry/config.yml
        source:
          secret: ${SECRET_NAME}
    hostNetworking: true
    context:
      priority: ${PRIORITY}
      minRuntime: ${MIN_RUNTIME}
      autoResume: true
    constraints:
      cluster:
        - ${CLUSTER}
EOF

echo "Creating registry mirror experiment from ${beaker_file}"
experiment_ref="$(beaker experiment create --workspace "${WORKSPACE}" "${beaker_file}" | awk 'NF {last=$NF} END {print last}')"
experiment_id="${experiment_ref##*/}"

if [[ -z "${experiment_id}" ]]; then
    echo "Failed to parse experiment id from Beaker output" >&2
    exit 1
fi

echo "Created experiment: ${experiment_id}"
echo "Renaming experiment to ${EXPERIMENT_NAME}"
beaker experiment rename "${experiment_id}" "${EXPERIMENT_NAME}"

cat <<EOF

Registry mirror launched.

Experiment: ${experiment_id}
Workspace:  ${WORKSPACE}
Cluster:    ${CLUSTER}
Port:       ${REGISTRY_PORT}

Use the task host/IP plus port ${REGISTRY_PORT} as MIRROR_URL for SWERL jobs.
Check the Beaker task logs to confirm incoming registry requests.
EOF
