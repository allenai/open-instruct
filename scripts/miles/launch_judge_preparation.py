"""CPU/WEKA preparation for the tiny multi-node judge exercise; always Saturn."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    document = launch.specification(args.image, spec)
    task = document["tasks"][0]
    task["name"] += "-prepare"
    for field in (
        "replicas",
        "leaderSelection",
        "hostNetworking",
        "propagateFailure",
        "propagatePreemption",
        "synchronizedStartTimeout",
    ):
        task.pop(field, None)
    task["resources"] = {"cpuCount": 8, "memory": "48 GiB"}
    task["constraints"]["cluster"] = ["ai2/saturn"]
    task["context"].update(minRuntime="20m", autoResume=False)
    task["timeout"] = "30m"
    # Keep the exact base64 frozen run; replace only the GPU preflight/entrypoint.
    command = task["arguments"][0]
    command = "\n".join(line for line in command.splitlines() if "preflight_attention" not in line)
    command = command.replace(
        "python -m open_instruct.miles.cluster /output/submitted-run.json",
        "python -m scripts.miles.prepare_judge_exercise /output/submitted-run.json",
    )
    task["arguments"] = [command]
    task["envVars"].append({"name": "LD_LIBRARY_PATH", "value": "/usr/local/cuda/compat"})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(json.dumps(document))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", spec.launch["workspace"], "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
