#!/bin/bash
set -euo pipefail
image="$1"
run_id="$2"
cluster="${3:-ai2/saturn}"
[[ "$cluster" == "ai2/saturn" || "$cluster" == "ai2/jupiter" ]]
spec="/tmp/h079-${run_id}-cache-spawn-smoke.json"
python scripts/train/atomic_think/prepare_spec.py "$image" "$run_id" "$cluster" "$spec"
python - "$spec" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
spec = json.loads(path.read_text())
spec['description'] = 'H079 full-cache spawned-worker qualification: actual Core loader, four workers, both arms; CPU only'
task = spec['tasks'][0]
task['name'] = 'cache-smoke'
task['command'] = ['python', 'scripts/train/atomic_think/cache_spawn_smoke.py']
root = '/weka/oe-training-default/robertb/atomic-think-h077'
task['arguments'] = [f'{root}/20261015d', f'{root}/20261016a']
task['timeout'] = '15m'
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h079-${run_id}-cache-spawn-smoke" --format json
