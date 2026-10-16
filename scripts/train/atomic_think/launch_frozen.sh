#!/bin/bash
set -euo pipefail
image="$1"
run_id="$2"
label="${3:-frozen}"
minutes="${4:-30}"
arms=(gate-reference gate-probed)
if (( $# > 4 )); then
    arms=("${@:5}")
fi
spec="/tmp/h077-${run_id}-${label}.json"
python scripts/train/atomic_think/spec.py --image "$image" --arm gate --run-id "$run_id" --minutes "$minutes" --output "$spec"
python - "$spec" "${arms[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H077 frozen checkpoint routing repeat; native eval-only, no optimizer or training'
task = spec['tasks'][0]
root = task['arguments'][1]
task['command'] = ['torchrun', '--standalone', '--nproc_per_node=1', 'scripts/train/atomic_think/frozen_probe.py']
task['arguments'] = [root, '--arms', *sys.argv[2:]]
task['resources']['gpuCount'] = 1
task.pop('replicas')
task.pop('leaderSelection')
task.pop('synchronizedStartTimeout')
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-${label}" --format json
