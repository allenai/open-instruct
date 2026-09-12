"""CPU/WEKA preparation, inspection or audit of the tiny judge exercise; always Saturn."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec


def specification(image, spec, stage):
    document = launch.specification(image, spec)
    task = document["tasks"][0]
    task["name"] += "-" + stage
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
    if stage == "prepare":
        entrypoint = "python -m scripts.miles.prepare_judge_exercise /output/submitted-run.json"
    elif stage == "audit":
        entrypoint = (
            "python -m scripts.miles.audit_judge_exercise "
            + shlex.quote(spec.output["root"])
            + " --report /output/audit.json"
        )
    elif stage == "inspect":
        # Read-only standard-library inspection can use the exact training image;
        # it needs no newly baked Python module or GPU allocation.
        source = f"""from pathlib import Path
import json
root = Path({spec.output["root"]!r})
for path in sorted((root / 'cluster').glob('*/*')):
    if path.suffix not in ('.json', '.log') or not path.is_file():
        continue
    raw = path.read_bytes()
    target = Path('/output/inspection') / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw[-2*1024*1024:])
    print(str(path), flush=True)
    print(raw[-16000:].decode(errors='replace'), flush=True)
"""
        entrypoint = "python -c " + shlex.quote(source)
    else:
        raise ValueError("Unknown CPU exercise stage")
    command = task["arguments"][0]
    command = "\n".join(line for line in command.splitlines() if "preflight_attention" not in line)
    command = command.replace("python -m open_instruct.miles.cluster /output/submitted-run.json", entrypoint)
    task["arguments"] = [command]
    task["envVars"].append({"name": "LD_LIBRARY_PATH", "value": "/usr/local/cuda/compat"})
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("prepare", "inspect", "audit"), default="prepare")
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    document = specification(args.image, spec, args.stage)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "experiment.json"
        path.write_text(json.dumps(document))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", spec.launch["workspace"], "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
