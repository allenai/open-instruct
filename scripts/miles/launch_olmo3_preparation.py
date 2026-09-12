"""Submit pinned Olmo 3 checkpoint/data preparation on Saturn, without GPUs."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec


def specification(image, spec):
    document = launch.specification(image, spec)
    if len(document["tasks"]) != 1:
        raise ValueError("Olmo 3 qualification preparation expects a single-node run config")
    task = document["tasks"][0]
    task["name"] += "-prepare"
    task["resources"] = {"cpuCount": 8, "memory": "48 GiB"}
    task["constraints"] = {"cluster": ["ai2/saturn"]}
    task["context"].update(minRuntime="20m", autoResume=False)
    task["timeout"] = "2h"
    command = "\n".join(line for line in task["arguments"][0].splitlines() if "preflight_attention" not in line)
    old = "python -m open_instruct.miles train /output/submitted-run.json"
    if old not in command:
        raise ValueError("Expected the single-node training entrypoint")
    task["arguments"] = [
        command.replace(old, "python -m scripts.miles.prepare_olmo3_checkpoint /output/submitted-run.json")
    ]
    task["envVars"].append({"name": "LD_LIBRARY_PATH", "value": "/usr/local/cuda/compat"})
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    document = specification(args.image, spec)
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
