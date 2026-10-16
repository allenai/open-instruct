#!/bin/bash
set -euo pipefail
image="$1"
run_id="$2"
cluster="${3:-ai2/saturn}"
python scripts/train/atomic_think/prepare_spec.py "$image" "$run_id" "$cluster" "/tmp/h077-${run_id}-prepare.json"
beaker experiment create "/tmp/h077-${run_id}-prepare.json" --workspace ai2/open-instruct-dev --name "h077-${run_id}-prepare" --format json
