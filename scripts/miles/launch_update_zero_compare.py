"""Read retained GPU traces in a bounded CPU-only Saturn comparison job."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1")
IMAGE = "01M26N80T0V9PREQTS87J849P8"


def specification(image, core_root, megatron_root, *, hf_only=False, evidence_only=False):
    if hf_only and evidence_only:
        raise ValueError("Choose one limited evidence mode")
    if image != IMAGE:
        raise ValueError("Use the original qualified Core runtime for the CPU comparison")
    roots = {"core": Path(core_root), "megatron": Path(megatron_root)}
    for backend, path in roots.items():
        if not path.is_relative_to(ROOT) or ".." in path.parts or path.name != backend:
            raise ValueError("Expected retained backend directory under the original GSM8K campaign")
    source = Path(__file__).with_name("compare_update_zero.py").read_bytes()
    encoded = base64.b64encode(source).decode()
    provenance = {
        "image": image,
        "hf_only": hf_only,
        "evidence_only": evidence_only,
        "compare_source_sha256": hashlib.sha256(source).hexdigest(),
        "backend_roots": {name: str(path) for name, path in roots.items()},
    }
    encoded_provenance = base64.b64encode(json.dumps(provenance, indent=2).encode()).decode()
    command = (
        "\n".join(
            [
                "set -euo pipefail",
                "mkdir -p /output",
                f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/compare_update_zero.py",
                f"printf %s {shlex.quote(encoded_provenance)} | base64 -d > /output/compare-launch.json",
                "export OMP_NUM_THREADS=8",
                "export MKL_NUM_THREADS=8",
                shlex.join(
                    [
                        "python",
                        "/tmp/compare_update_zero.py",
                        str(ROOT),
                        "--core-root",
                        str(roots["core"]),
                        "--megatron-root",
                        str(roots["megatron"]),
                        "--output",
                        "/output/comparison.json",
                    ]
                    + (["--hf-only"] if hf_only else [])
                    + (["--evidence-only"] if evidence_only else [])
                ),
            ]
        )
        + "\n"
    )
    return {
        "version": "v2",
        "description": "Read-only fixed-prefix update-zero tensor comparison: Core retry versus original Megatron",
        "tasks": [
            {
                "name": "compare-update-zero",
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"cpuCount": 8, "memory": "32 GiB", "sharedMemory": "4 GiB"},
                "context": {"priority": "urgent", "minRuntime": "20m", "autoResume": False},
                "constraints": {"cluster": ["ai2/saturn"]},
                "timeout": "30m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--core-root", default=str(ROOT / "update-zero-20260911-v2/core"))
    parser.add_argument("--megatron-root", default=str(ROOT / "update-zero-20260911-v1/megatron"))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--hf-only", action="store_true")
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()
    document = specification(
        args.image, args.core_root, args.megatron_root, hf_only=args.hf_only, evidence_only=args.evidence_only
    )
    if args.render_only:
        print(json.dumps(document, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="update-zero-compare-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
