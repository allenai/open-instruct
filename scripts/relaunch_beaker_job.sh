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

echo "Fetching experiment spec for ${EXPERIMENT_ID}..."
beaker experiment spec "$EXPERIMENT_ID" > "$YAML_PATH"

echo "Re-launching experiment..."
NEW_EXPERIMENT_ID=$(beaker experiment create "$YAML_PATH" --format json | jq -r '.id')
echo "New experiment launched: ${NEW_EXPERIMENT_ID}"
