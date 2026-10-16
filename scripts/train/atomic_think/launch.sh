#!/bin/bash
set -euo pipefail
image="$1"
arm="$2"
run_id="$3"
minutes="${4:-90}"
python scripts/train/atomic_think/spec.py --image "$image" --arm "$arm" --run-id "$run_id" --minutes "$minutes" --output "/tmp/h077-${run_id}-${arm}.json"
beaker experiment create "/tmp/h077-${run_id}-${arm}.json" --workspace ai2/open-instruct-dev --name "h077-${run_id}-${arm}" --format json
