#!/bin/bash
set -euo pipefail
image="$1"
run_id="$2"
cluster="${3:-ai2/saturn}"
spec="/tmp/h077-${run_id}-diagnose.json"
python scripts/train/atomic_think/prepare_spec.py "$image" "$run_id" "$cluster" "$spec"
python - "$spec" "${@:4}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H077 reproduce raw-dataset cache startup with stack dumps; shared CPU only'
task = spec['tasks'][0]
task['name'] = 'diagnose'
task['command'] = ['python', 'scripts/train/atomic_think/diagnose_startup.py']
task['arguments'].extend(sys.argv[2:])
# Match the training spec exactly: HF_HOME alone selects its datasets subfolder,
# whereas historical GPU launches explicitly use HF_HOME itself for Arrow data.
task['envVars'].append({'name': 'HF_DATASETS_CACHE', 'value': '/weka/oe-adapt-default/allennlp/.cache/huggingface'})
task['timeout'] = '10m'
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-diagnose" --format json
