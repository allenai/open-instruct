#!/usr/bin/env bash
# Relaunch an existing Beaker job by cloning its YAML spec.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <job-id>" >&2
  exit 1
fi

JOB_ID="$1"
WORKDIR="$(mktemp -d)"
YAML_PATH="$WORKDIR/job.yaml"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Fetching job spec for ${JOB_ID}..."
beaker job spec "$JOB_ID" > "$YAML_PATH"

echo "Re-launching job..."
NEW_JOB_ID=$(beaker job create "$YAML_PATH" | tail -n1)
echo "New job launched: ${NEW_JOB_ID}"
