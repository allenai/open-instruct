"""Render unallocated CPU preparation on Saturn, with explicit Jupiter fallback."""

import json
import sys
from pathlib import Path

image, run_id, cluster, output = sys.argv[1:]
assert cluster in ("ai2/saturn", "ai2/jupiter")
assert all(c.isalnum() or c in "-_" for c in run_id)
spec = {
    "version": "v2",
    "description": "H077 verify and copy historical packed-data indices; CPU only",
    "tasks": [
        {
            "name": "prepare",
            "image": {"beaker": image},
            "command": ["python", "scripts/train/atomic_think/prepare.py"],
            "arguments": [f"/weka/oe-training-default/robertb/atomic-think-h077/{run_id}"],
            "envVars": [
                {"name": "PYTHONPATH", "value": "/stage"},
                {"name": "HF_TOKEN", "secret": "robertb_HF_TOKEN"},
                {"name": "HF_HOME", "value": "/weka/oe-adapt-default/allennlp/.cache/huggingface"},
                {"name": "HF_HUB_CACHE", "value": "/weka/oe-adapt-default/allennlp/.cache/hub"},
                {"name": "OLMO_DATA_PREP_WORKERS", "value": "4"},
            ],
            "datasets": [
                {"mountPath": f"/weka/{name}", "source": {"weka": name}}
                for name in ("oe-adapt-default", "oe-training-default")
            ],
            "result": {"path": "/output"},
            "resources": {"cpuCount": 8, "memory": "64 GiB", "gpuCount": 0},
            "context": {"priority": "high", "autoResume": False},
            "constraints": {"cluster": [cluster]},
            "timeout": "30m",
        }
    ],
    "retry": {"allowedTaskRetries": 0},
}
Path(output).write_text(json.dumps(spec, indent=2) + "\n")
