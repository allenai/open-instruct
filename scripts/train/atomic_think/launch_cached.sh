#!/bin/bash
set -euo pipefail
image="$1"
arm="$2"
run_id="$3"
minutes="${4:-60}"
source_run="${5:-20261015d}"
cache_name="${6:-sample-cache-original}"
spec="/tmp/h077-${run_id}-${arm}-cached.json"
python scripts/train/atomic_think/spec.py --image "$image" --arm "$arm" --run-id "$run_id" --minutes "$minutes" --output "$spec"
python - "$spec" "$source_run" "$cache_name" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
assert all(value and all(c.isalnum() or c in '-_' for c in value) for value in sys.argv[2:])
source_root = f'/weka/oe-training-default/robertb/atomic-think-h077/{sys.argv[2]}'
spec['description'] = 'H077 fresh 512-update SFT arm; exact packed samples staged node-locally; offline probes'
task = spec['tasks'][0]
task['command'] = ['python', 'scripts/train/atomic_think/run_cached.py']
task['arguments'].extend([source_root, f'{source_root}/{sys.argv[3]}'])
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-${arm}-cached" --format json
