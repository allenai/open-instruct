#!/usr/bin/env bash
# Relaunch an existing Beaker experiment by cloning its YAML spec.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment-id>" >&2
  exit 1
fi

EXPERIMENT_ID="$1"
WORKDIR="$(mktemp -d)"
YAML_PATH="$WORKDIR/experiment.yaml"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Fetching experiment info for ${EXPERIMENT_ID}..."
EXPERIMENT_JSON=$(beaker experiment get "$EXPERIMENT_ID" --format json)
WORKSPACE=$(echo "$EXPERIMENT_JSON" | jq -r '.[0].workspaceRef.fullName')

echo "Fetching experiment spec..."
beaker experiment spec "$EXPERIMENT_ID" > "$YAML_PATH"

echo "Re-launching experiment in workspace ${WORKSPACE}..."
NEW_EXPERIMENT_ID=$(beaker experiment create "$YAML_PATH" --workspace "$WORKSPACE" --format json | jq -r '.[0].id')
echo "New experiment launched: ${NEW_EXPERIMENT_ID}"
