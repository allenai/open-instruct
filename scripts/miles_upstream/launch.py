"""Single-node Beaker submission for the upstream image and compiled native options."""

import base64
import copy
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from open_instruct.miles.configuration import validation
from open_instruct.miles.errors import InputError

IMAGE_ID = r"[0-9A-HJKMNP-TV-Z]{26}"
RESERVED_ENV = {
    "PYTHONPATH",
    "RAY_ADDRESS",
    "CUDA_VISIBLE_DEVICES",
    "MILES_UPSTREAM_BUILD_ID",
    "SGLANG_EXTERNAL_MODEL_PACKAGE",
}


def launch_settings(plan):
    native = plan["native"]
    launch = copy.deepcopy(plan["launch"])
    defaults = {
        "workspace": "ai2/open-instruct-dev",
        "budget": "ai2/oe-other",
        "cluster": "ai2/holmes",
        "priority": "high",
        "min_runtime": "0s",
        "timeout": "30m",
        "auto_resume": False,
        "shared_memory": "32 GiB",
        "env": {},
        "secrets": {},
        "weka_mounts": [],
    }
    validation.fields(launch, "launch", {*defaults, "gpus_per_replica"})
    launch = defaults | launch
    nodes = validation.integer(native["actor_num_nodes"], "trainer.trainer_num_nodes")
    if nodes != 1:
        raise InputError(
            "The upstream Beaker launcher currently supports one node; use a native upstream multi-node launch"
        )
    trainer_gpus = validation.integer(native["actor_num_gpus_per_node"], "trainer.gpus")
    if native.get("colocate", False):
        if native.get("fully_async", False):
            raise InputError("Fully asynchronous training needs disaggregated GPUs")
        if "rollout_num_gpus" in native:
            raise InputError(
                "inference.gpus is ignored upstream under colocation; omit it to avoid an ambiguous allocation"
            )
        required = trainer_gpus
    else:
        if "rollout_num_gpus" not in native:
            raise InputError("Disaggregated runs require explicit inference.gpus")
        required = trainer_gpus + validation.integer(native["rollout_num_gpus"], "inference.gpus")
    allocated = validation.integer(launch.get("gpus_per_replica", required), "launch.gpus_per_replica")
    if allocated != required or allocated > 8:
        raise InputError(f"Single-node launch needs exactly {required} GPUs and supports at most 8")
    launch["gpus_per_replica"] = allocated
    if native.get("num_gpus_per_node") != allocated:
        raise InputError(f"inference.rollout_gpus_per_node must equal the allocation ({allocated})")
    # Additional actor pools or external services need their own resource accounting.
    for key in ("use_critic", "eval_num_gpus", "rollout_external", "rollout_external_engine_addrs"):
        if native.get(key):
            raise InputError(
                f"miles.{key} needs a native launch; this launcher accounts for one trainer and rollout pool"
            )
    if native.get("cluster_backend", "ray") != "ray" or native.get("deploy_component", "all") != "all":
        raise InputError("The Beaker wrapper requires cluster_backend=ray and deploy_component=all")
    validation.choice(launch["priority"], "launch.priority", ("low", "normal", "high", "urgent"))
    for key in ("workspace", "budget", "cluster", "shared_memory"):
        validation.text(launch[key], f"launch.{key}")
    for key in ("min_runtime", "timeout"):
        seconds = validation.duration(launch[key], f"launch.{key}", allow_blank=key == "min_runtime")
        if not seconds:
            launch[key] = "0s"
    validation.boolean(launch["auto_resume"], "launch.auto_resume")
    if native.get("save_interval") is not None:
        validation.integer(native["save_interval"], "training.save_interval")
    if launch["auto_resume"] and (
        not native.get("save") or native.get("load") != native["save"] or not native.get("save_interval")
    ):
        raise InputError(
            "auto_resume requires miles.load to match output.root/checkpoints and a positive save_interval"
        )
    for group in ("env", "secrets"):
        validation.mapping(launch[group], f"launch.{group}")
        for key, value in launch[group].items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in RESERVED_ENV:
                raise InputError(f"Reserved or invalid launch.{group} variable: {key}")
            if not isinstance(value, str):
                raise InputError(f"launch.{group}.{key} must be a string")
            if group == "env" and key.upper().endswith(("_TOKEN", "_API_KEY", "_PASSWORD", "_SECRET")):
                raise InputError(f"Put {key} in launch.secrets")
    if set(launch["env"]) & set(launch["secrets"]):
        raise InputError("launch.env and launch.secrets must not overlap")
    if not isinstance(launch["weka_mounts"], list):
        raise InputError("launch.weka_mounts must be a list")
    for mount in launch["weka_mounts"]:
        validation.mapping(mount, "launch.weka_mounts entry")
        validation.fields(mount, "launch.weka_mounts entry", {"weka", "mount_path"})
        validation.text(mount.get("weka"), "weka")
        if not Path(validation.text(mount.get("mount_path"), "mount_path")).is_absolute():
            raise InputError("WEKA mount paths must be absolute")

    def check_mounts(value):
        if isinstance(value, list):
            for item in value:
                check_mounts(item)
        elif (
            isinstance(value, str)
            and value.startswith("/weka/")
            and not any(Path(value).is_relative_to(m["mount_path"]) for m in launch["weka_mounts"])
        ):
            raise InputError(f"Missing WEKA mount for {value}")

    for value in native.values():
        check_mounts(value)
    if launch["auto_resume"] and not any(
        Path(native["save"]).is_relative_to(m["mount_path"]) for m in launch["weka_mounts"]
    ):
        raise InputError("auto_resume checkpoints must be on a configured persistent WEKA mount")
    return launch


def specification(image, plan):
    launch = launch_settings(plan)
    frozen = {key: value for key, value in plan.items() if key != "launch"}
    payload = base64.b64encode(json.dumps(frozen).encode()).decode()
    unpack = f"import base64,pathlib; pathlib.Path('/output/run.json').write_bytes(base64.b64decode({payload!r}))"
    command = (
        "set -euo pipefail\nmkdir -p /output\n"
        f"python -c {shlex.quote(unpack)}\n"
        "cp /opt/miles-upstream/runtime.lock.json /output/runtime.lock.json\n"
        "trap 'status=$?; ray stop >/dev/null 2>&1 || true; exit $status' EXIT\n"
        f"ray start --head --num-gpus={launch['gpus_per_replica']} --include-dashboard=false\n"
        "python -m scripts.miles_upstream.execute /output/run.json 2>&1 | tee /output/run.log\n"
    )
    env = {"TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", **launch["env"]}
    return {
        "version": "v2",
        "budget": launch["budget"],
        "description": f"Upstream MILES: {plan['name']}",
        "tasks": [
            {
                "name": plan["name"],
                "image": {"beaker": image},
                "command": ["bash", "-c"],
                "arguments": [command],
                "datasets": [
                    {"mountPath": m["mount_path"], "source": {"weka": m["weka"]}} for m in launch["weka_mounts"]
                ],
                "result": {"path": "/output"},
                "resources": {"gpuCount": launch["gpus_per_replica"], "sharedMemory": launch["shared_memory"]},
                "context": {
                    "priority": launch["priority"],
                    "minRuntime": launch["min_runtime"],
                    "autoResume": launch["auto_resume"],
                },
                "constraints": {"cluster": [launch["cluster"]]},
                "timeout": launch["timeout"],
                "envVars": [{"name": k, "value": v} for k, v in env.items() if k not in launch["secrets"]]
                + [{"name": k, "secret": v} for k, v in launch["secrets"].items()],
            }
        ],
    }


def submit(image, plan, receipt_path):
    if not re.fullmatch(IMAGE_ID, image):
        raise InputError("--image must be an immutable Beaker image ID")
    document = specification(image, plan)
    metadata = json.loads(subprocess.check_output(["beaker", "image", "get", image, "--format", "json"], text=True))[0]
    if metadata["id"] != image or f"build-id={plan['build_id']}" not in metadata.get("description", ""):
        raise InputError(
            "Beaker image does not match the exported option schemas; use this builder's application image"
        )
    launch = launch_settings(plan)
    print(
        f"Launching on {launch['cluster']}: 1 node, {launch['gpus_per_replica']} GPUs, priority={launch['priority']}, "
        f"minRuntime={launch['min_runtime']}, autoResume={launch['auto_resume']}, timeout={launch['timeout']}"
    )
    with tempfile.TemporaryDirectory(prefix="miles-upstream-submit-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        response = subprocess.check_output(
            ["beaker", "experiment", "create", str(path), "--workspace", launch["workspace"], "--format", "json"],
            text=True,
        )
    experiment = json.loads(response)[0]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps({"image": image, "experiment": experiment, "specification": document}, indent=2) + "\n"
    )
    print(f"https://beaker.org/ex/{experiment['id']}")
    print(f"Receipt: {receipt_path}")
