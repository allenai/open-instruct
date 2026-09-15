"""Launch the frozen baseline preparation/training with committed source overlays."""

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path

from scripts.miles import launch_judge_preparation

from open_instruct.miles import launch
from open_instruct.miles.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[2]


def document(
    image, run, stage, source, digest, *, hostnames=None, prepare_module="scripts.miles.prepare_baseline_basket"
):
    spec = (
        launch_judge_preparation.specification(image, run, "prepare", prepare_module=prepare_module)
        if stage == "prepare"
        else launch.specification(image, run, hostnames=hostnames)
    )
    overlay = (
        "mkdir -p /output\ncp /qualification-source/provenance.json /output/\n"
        + f"echo '{digest}  /qualification-source/source.tar' | sha256sum -c -\n"
        + "tar -xf /qualification-source/source.tar -C /opt/core-rl\n"
        + "git -C /opt/core-rl/sources/miles apply --check /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch\n"
        + "git -C /opt/core-rl/sources/miles apply /opt/core-rl/scripts/miles/diagnostics/policy-refresh-runtime.patch\n"
    )
    for index, task in enumerate(spec["tasks"]):
        node_overlay = overlay
        if stage == "prepare":
            task["resources"].update(cpuCount=24, memory="96 GiB")
            task["timeout"] = "2h"
            if prepare_module == "scripts.miles.audit_live_trainer":
                task["resources"].update(gpuCount=0, cpuCount=2, memory="16 GiB")
                task["constraints"] = {"cluster": ["ai2/saturn"]}
                task["hostNetworking"] = True
                task["timeout"] = "10m"
                task["context"].update(minRuntime="5m")
            if prepare_module == "scripts.miles.qualify_judge_context":
                task["resources"].update(gpuCount=1, cpuCount=16, memory="128 GiB")
                task["constraints"] = {"cluster": ["ai2/holmes"]}
                task["context"].update(minRuntime="30m")
                task["timeout"] = "1h"
                task["envVars"] = [entry for entry in task["envVars"] if entry["name"] != "LD_LIBRARY_PATH"]
        else:
            target = Path(run.output["root"]) / f"checkpoints/gpu_usage_node{index}.jsonl"
            node_overlay += f"python -m scripts.miles.sample_gpu_usage {shlex.quote(str(target))} &\n"
            # Runtime IP sorting determines which replica owns the trainer. Both
            # reserve enough host RAM for its native optimizer checkpoint staging.
            task["resources"].update(cpuCount=48, memory="704 GiB", sharedMemory="200 GiB")
        task["datasets"].append(dict(mountPath="/qualification-source", source=dict(beaker=source)))
        task["arguments"][0] = task["arguments"][0].replace("cd /opt/core-rl\n", "cd /opt/core-rl\n" + node_overlay, 1)
        if stage == "train":
            task["arguments"][0] = task["arguments"][0].replace(
                "python -m open_instruct.miles.cluster /output/submitted-run.json",
                "python -m scripts.miles.prepare_baseline_basket /output/submitted-run.json --verify --wait-seconds 1800\n"
                "python -m open_instruct.miles.cluster /output/submitted-run.json",
            )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("prepare", "train", "workflow"), required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--prepare-module",
        choices=(
            "scripts.miles.prepare_baseline_basket",
            "scripts.miles.prepare_olmo3_basket",
            "scripts.miles.audit_judge_budget",
            "scripts.miles.audit_live_trainer",
            "scripts.miles.qualify_judge_context",
        ),
        default="scripts.miles.prepare_baseline_basket",
    )
    args = parser.parse_args()
    if args.image != "01M2CJG5RQQ93GEYNYAS7ASCQJ":
        parser.error(
            "This campaign overlay requires base image 01M2CJG5RQQ93GEYNYAS7ASCQJ; "
            "newer packaged images already contain a different MILES patch state."
        )
    run = RunSpec.load(args.config)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit changes before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="baseline-source-") as temporary:
        directory = Path(temporary)
        archive = directory / "source.tar"
        archive.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "archive",
                    "HEAD",
                    "open_instruct/miles",
                    "scripts/miles",
                    "tests/miles",
                    "configs/miles",
                    "runtime/miles/runtime.lock.json",
                ],
                cwd=ROOT,
            )
        )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        provenance = dict(commit=commit, base_image=args.image, archive_sha256=digest, stage=args.stage)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
        source = "SOURCE_DATASET"
        if not args.render_only:
            name = "baseline-source-" + uuid.uuid4().hex[:12]
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "create",
                    str(directory),
                    "--workspace",
                    run.launch["workspace"],
                    "--budget",
                    run.launch["budget"],
                    "--name",
                    name,
                ],
                check=True,
            )
            author = json.loads(
                subprocess.check_output(["beaker", "account", "whoami", "--format", "json"], text=True)
            )[0]["name"]
            source = json.loads(
                subprocess.check_output(
                    ["beaker", "dataset", "get", f"{author}/{name}", "--format", "json"], text=True
                )
            )[0]["id"]
        spec = document(
            args.image,
            run,
            args.stage,
            source,
            digest,
            hostnames=[f"host-{i}" for i in range(8)] if args.render_only else None,
            prepare_module=args.prepare_module,
        )
        if args.render_only:
            print(json.dumps(spec, indent=2))
            return
        path = directory / "experiment.json"
        path.write_text(json.dumps(spec))
        result = json.loads(
            subprocess.check_output(
                [
                    "beaker",
                    "experiment",
                    "create",
                    str(path),
                    "--workspace",
                    run.launch["workspace"],
                    "--format",
                    "json",
                ],
                text=True,
            )
        )[0]
        print(
            json.dumps(
                dict(
                    experiment=result["id"],
                    source=source,
                    output=run.output["root"],
                    allocation=run.plan()["allocation"],
                    spec=run.to_dict(),
                    **provenance,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
