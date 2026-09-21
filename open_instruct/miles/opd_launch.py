"""Single-allocation Beaker specification for OPD and CPU asset preparation."""

import base64
import json
import os
import shlex
import subprocess
from pathlib import Path

from open_instruct.miles.errors import InputError

# Directories the application image layer copies from the repository (runtime/miles/Dockerfile).
OVERLAY_DIRS = ("open_instruct", "scripts/miles", "configs/miles", "tests/miles")


def code_overlay():
    """Shell lines that replace the image's wrapper source with the committed HEAD, or ``""``.

    Set ``MILES_CODE_OVERLAY=1`` together with ``MILES_EXISTING_IMAGE`` when the runtime image
    cannot be rebuilt locally: the job fetches the exact HEAD commit (which must be pushed to
    ``origin``) and copies ``OVERLAY_DIRS`` over ``/opt/core-rl`` before ``open_instruct.miles``
    starts, so the code that runs matches the ``revision`` in the launch receipt. Runtime
    dependencies still come from the image; use a rebuilt image for dependency changes.
    """
    if os.environ.get("MILES_CODE_OVERLAY", "") not in ("1", "true"):
        return ""
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
    if origin.startswith("git@github.com:"):
        origin = "https://github.com/" + origin[len("git@github.com:") :]
    # The repository tracks test fixtures with git-lfs; the job has no LFS endpoint and does not
    # need them, and a failing smudge filter aborts the checkout (exit 128).
    lines = [
        "git init -q /tmp/oi-overlay",
        f"git -C /tmp/oi-overlay fetch -q --depth 1 {shlex.quote(origin)} {revision}",
        "GIT_LFS_SKIP_SMUDGE=1 git -C /tmp/oi-overlay checkout -q FETCH_HEAD",
        f"echo 'code overlay: {revision}'",
    ]
    lines += [f"cp -r /tmp/oi-overlay/{d}/. /opt/core-rl/{d}/" for d in OVERLAY_DIRS]
    return "".join(line + "\n" for line in lines)


def specification(image, spec):
    mounts = spec.launch["weka_mounts"]
    for key, value in spec.output.items():
        if value.startswith("/weka/") and not any(Path(value).is_relative_to(m["mount_path"]) for m in mounts):
            raise InputError(f"output.{key} requires a WEKA mount")
    for name in spec.launch["env"]:
        if name.upper().endswith(("_TOKEN", "_API_KEY", "_PASSWORD", "_SECRET")):
            raise InputError("Use launch.secrets for credentials")
    payload = base64.b64encode(json.dumps(spec.to_dict()).encode()).decode()
    setup = (
        f"import base64,pathlib; pathlib.Path('/output/submitted-run.json').write_bytes(base64.b64decode({payload!r}))"
    )
    collect = (
        f"from open_instruct.miles.launch import collect_results; collect_results({spec.output['root']!r}, '/output')"
    )
    cleanup = "status=$?; python -c " + shlex.quote(collect) + ' || true; exit "$status"'
    command = (
        "set -euo pipefail\ncd /opt/core-rl\nmkdir -p /output\n"
        + code_overlay()
        + f"python -c {shlex.quote(setup)}\ntrap {shlex.quote(cleanup)} EXIT\n"
        "python -m open_instruct.miles train /output/submitted-run.json 2>&1 | tee /output/run.log\n"
    )
    env = {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "HF_HOME": "/tmp/hf-cache",
        "WANDB_MODE": spec.document["tracking"]["wandb_mode"],
        "NCCL_CUMEM_ENABLE": "1",
        **spec.launch["env"],
    }
    task = {
        "name": spec.name,
        "image": {"beaker": image},
        "command": ["bash", "-c"],
        "arguments": [command],
        "datasets": [{"mountPath": m["mount_path"], "source": {"weka": m["weka"]}} for m in mounts],
        "result": {"path": "/output"},
        "resources": {
            "gpuCount": spec.allocation()["gpus_per_replica"],
            "memory": "256 GiB",
            "sharedMemory": spec.launch["shared_memory"],
        },
        "context": {
            "priority": spec.launch["priority"],
            "minRuntime": spec.launch["min_runtime"],
            "autoResume": spec.launch["auto_resume"],
        },
        "constraints": {"cluster": [spec.launch["cluster"]]},
        "timeout": spec.launch["timeout"],
        "envVars": [{"name": k, "value": v} for k, v in env.items() if k not in spec.launch["secrets"]]
        + [{"name": k, "secret": v} for k, v in spec.launch["secrets"].items()],
    }
    return {
        "version": "v2",
        "budget": spec.launch["budget"],
        "description": f"Miles Megatron OPD: {spec.name}",
        "tasks": [task],
    }
