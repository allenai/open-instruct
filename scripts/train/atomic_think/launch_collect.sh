#!/bin/bash
set -euo pipefail
image="$1"
run_id="$2"
label="$3"
shift 3
spec="/tmp/h077-${run_id}-collect.json"
python scripts/train/atomic_think/prepare_spec.py "$image" "$run_id" ai2/saturn "$spec"
python - "$spec" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H077 copy completed diagnostic artifacts from WEKA; shared CPU only'
task = spec['tasks'][0]
task['name'] = 'collect'
task['command'] = ['python', 'scripts/train/atomic_think/collect.py']
task['arguments'].extend(sys.argv[2:])
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-collect-${label}" --format json
