#!/bin/bash
set -euo pipefail

# Entry point for scripts/train/build_image_and_launch.sh --cuda-version 13.
image="${1:?Pass the built Beaker image}"
commit=$(git rev-parse HEAD)
description=$(beaker image get "$image" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["description"])')
if [[ "$description" != *"${commit:0:9}"* ]]; then
    echo "Image/source commit mismatch: $description; source $commit. Rebuild before launch." >&2
    exit 1
fi
python3 - "$image" "$commit" <<'PY' > /tmp/open-instruct-1939-parity-spec.json
import hashlib
import json
import sys
from pathlib import Path

image, commit = sys.argv[1:]
paths = ['scripts/test/sft_loss_parity.py', 'scripts/test/run_sft_loss_parity.sh',
         'open_instruct/sft_loss_utils.py', 'open_instruct/finetune.py']
hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
spec = {
    'version': 'v2', 'budget': 'ai2/oe-other',
    'description': f'PR1939 real ZeRO-2/3, SP2/DP2, fused loss and checkpoint parity; commit {commit}',
    'tasks': [{
        'name': 'sft-loss-gpu-parity', 'image': {'beaker': image},
        'command': ['/bin/bash', '-euo', 'pipefail', '-c'],
        'arguments': ['cd /stage && bash scripts/test/run_sft_loss_parity.sh'],
        'envVars': [{'name': k, 'value': v} for k, v in {
            'GIT_COMMIT': commit, 'EXPECTED_SOURCE_HASHES': json.dumps(hashes),
            'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MAX_JOBS': '4',
            'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled',
        }.items()],
        'resources': {'gpuCount': 4, 'cpuCount': 16, 'memory': '64 GiB', 'sharedMemory': '16 GiB'},
        'context': {'priority': 'high', 'minRuntime': '5m', 'autoResume': False},
        'constraints': {'cluster': ['ai2/holmes']},
        'timeout': '45m', 'result': {'path': '/output'},
    }],
}
assert image.endswith('-cuda13'), 'Holmes validation must use the pinned CUDA13 runtime'
print(json.dumps(spec, indent=2))
PY
beaker experiment create /tmp/open-instruct-1939-parity-spec.json \
    --workspace ai2/open-instruct-dev --name "sft-loss-parity-${commit:0:8}" \
    --format json > /tmp/open-instruct-1939-parity-created.json
cat /tmp/open-instruct-1939-parity-created.json
