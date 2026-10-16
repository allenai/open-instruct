#!/bin/bash
set -euo pipefail
image="$1"
arm="$2"
run_id="$3"
manifest_sha256="$4"
resume="$5"
minutes="${6:-420}"
spec="/tmp/h077-${run_id}-${arm}-long.json"
python scripts/train/atomic_think/spec.py --image "$image" --arm "$arm" --run-id "$run_id" --minutes 60 --output "$spec"
python - "$spec" "$arm" "$manifest_sha256" "$resume" "$minutes" <<'PY'
import json
import re
import sys
from pathlib import Path

path, arm, digest, resume, minutes = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
assert arm in ('control', 'atomic-safe')
assert re.fullmatch(r'[0-9a-f]{64}', digest)
assert resume.is_relative_to('/weka/oe-training-default/robertb/atomic-think-h077')
assert resume.parent.name == arm and re.fullmatch(r'step\d+', resume.name)
assert 0 < minutes <= 420
kind = 'atomic' if arm == 'atomic-safe' else 'original'
spec = json.loads(path.read_text())
spec['description'] = 'H077 matched SFT continuation to 11768 updates; fail-closed native restore; bounded offline audits'
task = spec['tasks'][0]
root = task['arguments'][1]
task['command'] = ['python', 'scripts/train/atomic_think/run_long.py']
task['arguments'] = [arm, root,
    '/weka/oe-training-default/robertb/atomic-think-h077/20261015d',
    f'/weka/oe-training-default/robertb/atomic-think-h077/20261016a/full-sample-cache-{kind}',
    str(resume), '--manifest-sha256', digest]
task['context']['minRuntime'] = f'{minutes}m'
task['timeout'] = f'{minutes}m'
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
beaker experiment create "$spec" --workspace ai2/open-instruct-dev --name "h077-${run_id}-${arm}-long" --format json
