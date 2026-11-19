#!/bin/bash
# Committed-source benchmark submission using an immutable, cluster-built runtime.
set -euo pipefail
if [[ $# != 1 || -n "$(git status --porcelain)" ]]; then
  echo 'Usage from a clean committed checkout: MILES_EXISTING_IMAGE=ID build_image_and_launch.sh --miles-benchmark SPEC'
  exit 1
fi
: "${MILES_EXISTING_IMAGE:?An immutable benchmark image ID is required}"
python - "$1" "$MILES_EXISTING_IMAGE" <<'PY'
import json, subprocess, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
image = json.loads(subprocess.check_output(['beaker', 'image', 'get', sys.argv[2], '--format', 'json']))[0]
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert image['id'] == sys.argv[2] and revision in image['description']
assert len(spec['tasks']) == 1 and spec['tasks'][0]['resources']['gpuCount'] == 1
assert spec['tasks'][0]['image']['beaker'] == image['id']
PY
exec beaker experiment create "$1" --workspace ai2/olmo-instruct --format json
