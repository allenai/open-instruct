#!/bin/bash
set -euo pipefail
image="$1"
arm="$2"
run_id="$3"
dataset="$4"
manifest_sha="$5"
qualify="${6:-no}"
spec="/tmp/h096-${run_id}.json"
python scripts/train/atomic_think/spec.py --image "$image" --arm "$arm" --run-id "$run_id" --minutes 90 --output "$spec"
python - "$spec" "$dataset" "$manifest_sha" "$qualify" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
assert re.fullmatch(r'[0-9a-f]{64}', sys.argv[3])
assert sys.argv[4] in ('yes', 'no')
spec['description'] = 'H096 warmed initialization: precision/resume qualification' if sys.argv[4] == 'yes' else 'H096 separate warmed initialization SFT treatment to step512'
spec['budget'] = '01KP6SET80STGTCC9EFRPAE2R6'
task = spec['tasks'][0]
task['command'] = ['python', 'scripts/train/atomic_think/run_warmed.py']
task['arguments'].extend(['/warmup/manifest.json', sys.argv[3]])
task['datasets'].append({'mountPath': '/warmup', 'source': {'beaker': sys.argv[2]}})
if sys.argv[4] == 'yes':
    task['arguments'].append('--qualify')
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h096-${run_id}" --format json
