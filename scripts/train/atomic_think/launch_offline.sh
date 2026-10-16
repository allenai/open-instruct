#!/bin/bash
set -euo pipefail
image="$1"
arm="$2"
run_id="$3"
label="$4"
minutes="$5"
shift 5
spec="/tmp/h077-${run_id}-${arm}-${label}.json"
python scripts/train/atomic_think/spec.py --image "$image" --arm "$arm" --run-id "$run_id" --minutes "$minutes" --output "$spec"
python - "$spec" "$arm" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H077 offline native-checkpoint probes; no trainer or optimizer'
task = spec['tasks'][0]
root = task['arguments'][1]
task['command'] = ['torchrun', '--standalone', '--nproc_per_node=1', 'scripts/train/atomic_think/offline_probe.py']
task['arguments'] = [root, sys.argv[2], *sys.argv[3:]]
task['resources']['gpuCount'] = 1
task.pop('replicas')
task.pop('leaderSelection')
task.pop('synchronizedStartTimeout')
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-${arm}-${label}" --format json
