"""Launch a committed, isolated two-GPU request refresh probe on Holmes."""

import base64
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = "/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf"


def make_spec(image, mode, radix=False, long_repeats=0):
    if mode not in ("tiny", "sft"):
        raise ValueError("Probe mode must be tiny or sft")
    if type(long_repeats) is not int or not 0 <= long_repeats <= 5:
        raise ValueError("long_repeats must be an integer from 0 to 5")
    files = {
        name: Path(__file__).with_name(name).read_bytes()
        for name in ("policy_refresh_probe.py", "policy_refresh_hooks.py")
    }
    provenance = {
        "image": image,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    commands = ["set -euo pipefail", "mkdir -p /output /tmp/policy-refresh", "cd /opt/core-rl"]
    files["provenance.json"] = json.dumps(provenance, indent=2).encode()
    for name, content in files.items():
        target = "/output/provenance.json" if name == "provenance.json" else "/tmp/policy-refresh/" + name
        commands.append(
            f"printf %s {shlex.quote(base64.b64encode(content).decode())} | base64 -d > {shlex.quote(target)}"
        )
    # Hook is probe-only and injected into this ephemeral container. Production
    # source pins/images are untouched; exact source bytes are in provenance.
    patch = "\nfrom policy_refresh_hooks import install as _install_refresh_probe\n_install_refresh_probe(Scheduler)\n"
    commands.append(
        f"printf %s {shlex.quote(base64.b64encode(patch.encode()).decode())} | base64 -d >> /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"
    )
    commands.append('export PYTHONPATH="/tmp/policy-refresh:$PYTHONPATH"')
    argv = ["python", "/tmp/policy-refresh/policy_refresh_probe.py"]
    if mode == "sft":
        argv += ["--model", MODEL]
    if radix:
        argv += ["--radix"]
    if long_repeats:
        argv += ["--long-repeats", str(long_repeats)]
    commands.append(shlex.join(argv))
    task = {
        "name": f"policy-refresh-{mode}-{'radix' if radix else 'no-radix'}",
        "image": {"beaker": image},
        "command": ["bash", "-lc"],
        "arguments": ["\n".join(commands)],
        "resources": {"cpuCount": 16, "gpuCount": 2, "memory": "192 GiB", "sharedMemory": "16 GiB"},
        "constraints": {"cluster": ["ai2/holmes"]},
        "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
        "result": {"path": "/output"},
        "timeout": "1h",
        "envVars": [{"name": "OMP_NUM_THREADS", "value": "2"}, {"name": "NCCL_CUMEM_ENABLE", "value": "1"}],
    }
    if mode == "tiny":
        task["resources"].update(cpuCount=8, memory="32 GiB", sharedMemory="4 GiB")
        task["context"]["minRuntime"] = "10m"
        task["timeout"] = "20m"
    if mode == "sft":
        task["datasets"] = [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}]
    return {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": "Live KDA pause, GPU weight update, re-prefill and continuation probe",
        "tasks": [task],
    }


def main():
    image = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    repeats = 3 if "--repeat-long" in sys.argv[3:] else 0
    spec = make_spec(image, mode, "--radix" in sys.argv[3:], long_repeats=repeats)
    with tempfile.TemporaryDirectory(prefix="policy-refresh-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(spec))
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
