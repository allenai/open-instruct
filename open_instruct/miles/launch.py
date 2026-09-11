"""Beaker submission and receipts for committed researcher run files."""

import base64
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles import workflow
from open_instruct.miles.run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[2]


def receipt_path(spec):
    directory = Path(os.environ.get("MILES_LAUNCH_RECEIPTS", Path.home() / ".cache/open-instruct/miles/launches"))
    key = workflow.fingerprint({"name": spec.name, "root": spec.output["root"]})[:16]
    return directory / f"{spec.name}-{key}.json"


def specification(image, spec):
    config = spec.compile()
    miles = config.miles
    trainer = miles["actor_num_nodes"] * miles["actor_num_gpus_per_node"]
    allocated = trainer if miles["colocate"] else trainer + miles["rollout_num_gpus"]
    capacity = spec.launch.get("gpus_per_replica", miles["num_gpus_per_node"])
    if miles["actor_num_nodes"] != 1 or allocated > capacity:
        raise ValueError(
            "The config launcher currently supports one Beaker node; this topology requires a multi-node "
            "Ray launcher. Use a one-node example or the existing qualified campaign launcher."
        )
    mounts = spec.launch["weka_mounts"]

    def check_mounts(value, name="run"):
        if isinstance(value, dict):
            for key, child in value.items():
                check_mounts(child, f"{name}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_mounts(child, f"{name}[{index}]")
        elif (
            isinstance(value, str)
            and value.startswith("/weka/")
            and not any(Path(value).is_relative_to(m["mount_path"]) for m in mounts)
        ):
            raise ValueError(f"{name} requires a corresponding launch.weka_mounts entry")

    check_mounts(spec.to_dict())
    sensitive = [
        name
        for name in spec.launch["env"]
        if name.upper().endswith(("_TOKEN", "_API_KEY", "_PASSWORD", "_SECRET", "_PRIVATE_KEY", "_ACCESS_KEY"))
    ]
    if sensitive:
        raise ValueError(f"Use launch.secrets for credential environment variables: {sorted(sensitive)}")
    payload = base64.b64encode(json.dumps(spec.to_dict()).encode()).decode()
    setup = (
        f"import base64,pathlib; pathlib.Path('/output/submitted-run.json').write_bytes(base64.b64decode({payload!r}))"
    )
    collect = (
        f"from open_instruct.miles.launch import collect_results; collect_results({spec.output['root']!r}, '/output')"
    )
    preflight = ""
    if config.core.attention_backend in ("torch", "flash_4"):
        preflight = f"python -m scripts.miles.preflight_attention --backend {config.core.attention_backend}\n"
    cleanup = "status=$?; python -c " + shlex.quote(collect) + ' || true; exit "$status"'
    command = (
        "set -euo pipefail\ncd /opt/core-rl\nmkdir -p /output\n"
        f"python -c {shlex.quote(setup)}\n"
        f"trap {shlex.quote(cleanup)} EXIT\n"
        + preflight
        + "python -m open_instruct.miles train /output/submitted-run.json 2>&1 | tee /output/run.log\n"
    )
    env = {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "NCCL_CUMEM_ENABLE": "1",
        "HF_HOME": "/tmp/hf-cache",
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models",
        **spec.launch["env"],
    }
    task = dict(
        name=spec.name,
        image={"beaker": image},
        command=["bash", "-c"],
        arguments=[command],
        datasets=[{"mountPath": mount["mount_path"], "source": {"weka": mount["weka"]}} for mount in mounts],
        result={"path": "/output"},
        resources={"gpuCount": allocated, "memory": "256 GiB", "sharedMemory": spec.launch["shared_memory"]},
        context={
            "priority": spec.launch["priority"],
            "minRuntime": spec.launch["min_runtime"],
            "autoResume": spec.launch["auto_resume"],
        },
        constraints={"cluster": [spec.launch["cluster"]]},
        timeout=spec.launch["timeout"],
        envVars=[{"name": key, "value": value} for key, value in env.items() if key not in spec.launch["secrets"]]
        + [{"name": key, "secret": value} for key, value in spec.launch["secrets"].items()],
    )
    return dict(
        version="v2", budget=spec.launch["budget"], description=f"MILES/Core researcher run: {spec.name}", tasks=[task]
    )


def collect_results(root, destination):
    """Keep small provenance/metric artifacts in Beaker; checkpoints stay on WEKA."""
    root, destination = Path(root), Path(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Result destination must be outside the run directory")
    copied = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Checkpoint tensor trees and HF descriptors can contain tens of
        # thousands of files. Retain their parent completion manifests only.
        dirs[:] = sorted(
            name for name in dirs if name not in {"hf", "export-hf", "model"} and not name.startswith(".")
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or path.suffix not in (".json", ".jsonl") or path.stat().st_size > 32 * 1024 * 1024:
                continue
            relative = path.relative_to(root)
            target = destination / "run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            copied.append(str(relative))
    return copied


def run(path, overrides):
    spec = RunSpec.load(path, overrides)
    # Validate launch feasibility before spending time building an image.
    specification("pending-build", spec)
    with tempfile.TemporaryDirectory(prefix="miles-submitted-run-") as directory:
        frozen = Path(directory) / "run.json"
        workflow.write_json(frozen, spec.to_dict())
        command = [
            "bash",
            "./scripts/train/build_image_and_launch.sh",
            "--miles",
            "scripts/train/debug/miles_workflow.sh",
            str(frozen),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def submit(image, spec):
    # Resolve aliases before submission so the receipt and task name one image.
    images = json.loads(subprocess.check_output(["beaker", "image", "get", image, "--format", "json"], text=True))
    metadata = images[0] if isinstance(images, list) else images
    resolved_image = metadata["id"]
    document = specification(resolved_image, spec)
    # Preserve the exact submitted source, parameters, and image in the receipt.
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="miles-workflow-") as directory:
        path = Path(directory) / "experiment.json"
        workflow.write_json(path, document)
        response = subprocess.check_output(
            ["beaker", "experiment", "create", str(path), "--workspace", spec.launch["workspace"], "--format", "json"],
            text=True,
        )
    experiments = json.loads(response)
    experiment = experiments[0] if isinstance(experiments, list) else experiments
    receipt = dict(
        experiment_id=experiment["id"],
        image=resolved_image,
        requested_image=image,
        revision=revision,
        spec_sha256=workflow.fingerprint(spec.to_dict()),
        spec=spec.to_dict(),
    )
    target = receipt_path(spec)
    if target.exists():
        old = json.loads(target.read_text())
        workflow.write_json(target.with_name(f"{target.stem}-{old['experiment_id']}.json"), old)
    workflow.write_json(target, receipt)
    print(
        json.dumps({**receipt, "receipt": str(target), "url": f"https://beaker.org/ex/{experiment['id']}"}, indent=2)
    )
    return receipt


def status(spec):
    target = receipt_path(spec)
    if not target.is_file():
        raise FileNotFoundError(f"No launch receipt: {target}")
    receipt = json.loads(target.read_text())
    response = json.loads(
        subprocess.check_output(
            ["beaker", "experiment", "get", receipt["experiment_id"], "--format", "json"], text=True
        )
    )
    experiment = response[0] if isinstance(response, list) else response
    # Return all attempts in chronological order; an older preempted job is not the current status.
    jobs = list(experiment.get("jobs", []))
    if not jobs:
        jobs = [job for task in experiment.get("tasks", []) for job in task.get("jobs", [])]
    jobs.sort(key=lambda job: (job.get("status", {}).get("created", job.get("created", "")), job.get("id", "")))
    return dict(
        receipt=str(target),
        experiment_id=receipt["experiment_id"],
        url=f"https://beaker.org/ex/{receipt['experiment_id']}",
        config_matches_submission=receipt["spec_sha256"] == workflow.fingerprint(spec.to_dict()),
        latest_job=jobs[-1] if jobs else None,
        attempts=jobs,
        experiment=experiment,
    )
