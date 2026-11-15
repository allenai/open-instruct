"""Freeze the small serving hook independently of training source overlays."""

import base64
import hashlib
from pathlib import Path

from open_instruct.miles.evaluation import evaluation_runner


def build(policy):
    execution = Path(__file__).resolve().parents[1] / "execution"
    root = execution.parents[2]
    raw = Path(policy["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["sha256"]:
        raise ValueError("Evaluator KDA policy differs from the configured hash")
    sources = {
        "oi_miles_serving/__init__.py": b"",
        "oi_miles_serving/kda_dispatch.py": (execution / "kda_dispatch.py").read_bytes(),
        "oi_miles_serving/quantized_publication.py": (execution / "quantized_publication.py").read_bytes(),
        "oi_miles_serving/evaluator_startup_budget.py": (
            root / "scripts/miles/evaluator_startup_budget.py"
        ).read_bytes(),
        "runtime.lock.json": (root / "runtime/miles/runtime.lock.json").read_bytes(),
        "policy.json": raw,
    }
    artifact = {
        "version": 1,
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "base64": base64.b64encode(data).decode()}
            for name, data in sources.items()
        },
    }
    evaluation_runner.serving_files(artifact)
    return artifact
