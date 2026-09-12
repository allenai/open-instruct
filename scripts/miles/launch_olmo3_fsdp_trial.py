"""Qualify two-GPU FSDP diagnostics before the full Olmo 3 GSM8K workflow."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec


def specification(image, spec, *, probe_only=False):
    document = launch.specification(image, spec)
    if len(document["tasks"]) != 1 or document["tasks"][0]["resources"]["gpuCount"] < 2:
        raise ValueError("The FSDP qualification requires one node with at least two GPUs")
    task = document["tasks"][0]
    entrypoint = "python -m open_instruct.miles train /output/submitted-run.json"
    if entrypoint not in task["arguments"][0]:
        raise ValueError("Expected the single-node training entrypoint")
    preflight = (
        "torchrun --standalone --nproc_per_node=2 -m pytest -q -s "
        "tests/miles/test_parameter_probe.py 2>&1 | tee /output/fsdp-probe.log\n"
    )
    if probe_only:
        task["name"] += "-probe"
        task["arguments"] = ["set -euo pipefail\ncd /opt/core-rl\nmkdir -p /output\n" + preflight]
        task["resources"] = {"gpuCount": 2, "memory": "32 GiB", "sharedMemory": "8 GiB"}
        task["envVars"] = [{"name": "OMP_NUM_THREADS", "value": "2"}]
        task.pop("datasets", None)
        task["context"].update(minRuntime="10m", autoResume=False)
        task["timeout"] = "20m"
    else:
        task["arguments"][0] = task["arguments"][0].replace(entrypoint, preflight + entrypoint)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--probe-only", action="store_true", help="Run the bounded two-GPU regression without training"
    )
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    images = json.loads(subprocess.check_output(["beaker", "image", "get", args.image, "--format", "json"], text=True))
    document = specification(images[0]["id"], spec, probe_only=args.probe_only)
    if args.render_only:
        print(json.dumps(document, indent=2))
        return
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(json.dumps(document))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", spec.launch["workspace"], "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
