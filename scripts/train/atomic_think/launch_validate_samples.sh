#!/bin/bash
set -euo pipefail
image="$1"
kind="$2"
cluster="${3:-ai2/saturn}"
spec="/tmp/h077-validate-samples-${kind}.json"
python scripts/train/atomic_think/prepare_spec.py "$image" 20261015d "$cluster" "$spec"
python - "$spec" "$kind" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H077 verify node-local sample cache against recorded training batches; CPU only'
task = spec['tasks'][0]
task['command'] = ['python', 'scripts/train/atomic_think/validate_samples.py']
task['arguments'].append(sys.argv[2])
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-validate-samples-${kind}" --format json
