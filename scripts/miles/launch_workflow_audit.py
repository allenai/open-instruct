"""Read-only CPU audit of a completed researcher run's WEKA artifacts."""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def specification(image, root, source):
    encoded = base64.b64encode(source).decode()
    setup = (
        f"import base64,pathlib; pathlib.Path('/output/audit_workflow.py').write_bytes(base64.b64decode({encoded!r}))"
    )
    command = (
        "set -euo pipefail\ncd /opt/core-rl\nmkdir -p /output\n"
        "export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 WANDB_MODE=disabled\n"
        "export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}\n"
        f"python -c {shlex.quote(setup)}\n"
        f"python /output/audit_workflow.py {shlex.quote(str(root))} --report /output/audit.json\n"
    )
    return dict(
        version="v2",
        budget="ai2/oe-other",
        description=f"Independent read-only config workflow audit; auditor {hashlib.sha256(source).hexdigest()}",
        tasks=[
            dict(
                name="workflow-audit",
                image={"beaker": image},
                command=["bash", "-c"],
                arguments=[command],
                datasets=[{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                result={"path": "/output"},
                resources={"cpuCount": 4, "memory": "32 GiB", "gpuCount": 0},
                constraints={"cluster": ["ai2/saturn"]},
                context={"priority": "urgent", "minRuntime": "10m", "autoResume": False},
                timeout="30m",
            )
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("root", type=Path)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.root.is_relative_to("/weka/oe-training-default"):
        parser.error("This bounded audit launcher mounts oe-training-default only")
    source = Path(__file__).with_name("audit_workflow.py").read_bytes()
    document = json.dumps(specification(args.image, args.root, source), indent=2)
    if args.render_only:
        print(document)
        return
    with tempfile.TemporaryDirectory(prefix="workflow-audit-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(document)
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
