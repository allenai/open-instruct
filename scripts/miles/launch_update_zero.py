"""Launch original Core100 or Megatron100 with before/after-publication probes."""

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from scripts.miles import update_zero_megatron

ROOT = "/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1"
IMAGES = {"core": "01M26N80T0V9PREQTS87J849P8", "megatron": update_zero_megatron.IMAGE}


def specification(image, backend, *, campaign="update-zero-20260911-v1", mode="original"):
    if backend not in IMAGES or image != IMAGES[backend]:
        raise ValueError("The selected backend requires its immutable original 100-update image")
    if not re.fullmatch(r"update-zero-[A-Za-z0-9_-]+", campaign):
        raise ValueError("Use a distinct update-zero campaign directory")
    if mode not in ("original", "hf-matched"):
        raise ValueError("Unknown diagnostic mode")
    source = Path(__file__).parent
    output = ROOT + "/" + campaign + "/" + backend
    probe_dir = "/tmp/zero-probe"
    files = {name: (source / name).read_text() for name in ("update_zero_driver.py", "update_zero_capture.py")}
    files["inputs.json"] = (source.parents[1] / "configs/miles/reference/gsm8k-update-zero-inputs.json").read_text()
    files["sitecustomize.py"] = (
        "import runpy\n"
        "runpy.run_path('/usr/lib/python3.12/sitecustomize.py')\n"
        "from update_zero_capture import install_import_hook\n"
        "install_import_hook()\n"
    )
    runtime_env = {}
    provenance = {"image": image, "backend": backend, "mode": mode, "inputs": json.loads(files["inputs.json"])}
    if backend == "megatron":
        manifest = json.loads((source / "diagnostics/update-zero-megatron.json").read_text())
        prepared = update_zero_megatron.prepare(
            manifest, driver=probe_dir + "/update_zero_driver.py", output=output, probe_dir=probe_dir
        )
        runtime_env = prepared["runtime_env"]
        command = prepared["command"]
        provenance["original_recipe"] = manifest
        provenance["diagnostic_recipe"] = prepared
        working_directory = "/root/miles"
    else:
        command = ["python", probe_dir + "/update_zero_driver.py"]
        working_directory = "/opt/core-rl"
        provenance["original_recipe"] = json.loads((source / "frozen_core_score_manifest.json").read_text())
    env = dict(runtime_env.get("env_vars", {}))
    env.update(
        OI_UPDATE_ZERO_ROOT=ROOT,
        OI_UPDATE_ZERO_MODE=mode,
        OI_UPDATE_ZERO_BACKEND=backend,
        OI_UPDATE_ZERO_OUTPUT=output,
        OI_UPDATE_ZERO_TRACE_DIR=output + "/trace",
        OI_UPDATE_ZERO_INPUTS=probe_dir + "/inputs.json",
        OI_UPDATE_ZERO_RAY_ENV=json.dumps(runtime_env),
        SGLANG_EXTERNAL_MODEL_PACKAGE="olmo_sglang.models",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        NCCL_CUMEM_ENABLE="1",
        HF_HOME="/tmp/hf-cache",
        PYTHONUNBUFFERED="1",
    )
    if mode == "hf-matched":
        caches = {
            "TRITON_CACHE_DIR": probe_dir + "/compiler-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": probe_dir + "/compiler-cache/inductor",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": probe_dir + "/compiler-cache/fa4",
            "TILELANG_CACHE_DIR": probe_dir + "/compiler-cache/tilelang",
            "EP_JIT_CACHE_DIR": probe_dir + "/compiler-cache/deepep",
            "DG_JIT_CACHE_DIR": probe_dir + "/compiler-cache/deep-gemm",
        }
        env.update(caches)
        runtime_env.setdefault("env_vars", {}).update(caches)
        env["OI_UPDATE_ZERO_RAY_ENV"] = json.dumps(runtime_env)
        provenance["fresh_compiler_caches"] = caches
    provenance["diagnostic_file_sha256"] = {
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()
    }
    files["manifest.json"] = json.dumps(provenance, indent=2)
    files["collect.py"] = (
        "import os,shutil\nfrom pathlib import Path\n"
        "root=Path(os.environ['OI_UPDATE_ZERO_OUTPUT']); dest=Path('/output')\n"
        "for p in root.rglob('*.json'):\n"
        " q=dest/p.relative_to(root); q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)\n"
    )
    lines = ["set -euo pipefail", f"cd {shlex.quote(working_directory)}", f"test ! -e {shlex.quote(output)}"]
    if mode == "hf-matched":
        lines.append(f"test ! -e {probe_dir}/compiler-cache")
    lines.append(f"mkdir -p {shlex.quote(output + '/trace')} {probe_dir} /output")
    for name, contents in files.items():
        encoded = base64.b64encode(contents.encode()).decode()
        lines.append(f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(probe_dir + '/' + name)}")
    for name, value in env.items():
        lines.append(f"export {name}={shlex.quote(value)}")
    if backend == "core":
        lines.append('export PYTHONPATH="/tmp/zero-probe:$PYTHONPATH"')
    lines.append(f"cp {probe_dir}/manifest.json {shlex.quote(output + '/manifest.json')}")
    lines.append(f"trap 'python {probe_dir}/collect.py' EXIT")
    if backend == "megatron":
        lines.append(prepared["bootstrap_script"])
    lines.append(
        "python -c 'import torch; assert \"B300\" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name(), torch.version.cuda)'"
    )
    lines.append(shlex.join(command))
    return {
        "version": "v2",
        "description": f"Original {backend}100 update-zero ({mode}): identical prefixes, expert IDs/activations; zero updates",
        "tasks": [
            {
                "name": "update-zero-" + backend,
                "image": {"beaker": image},
                "envVars": [{"name": "GITHUB_TOKEN", "secret": "robertb_GITHUB_TOKEN"}]
                if backend == "megatron"
                else [],
                "command": ["bash", "-c"],
                "arguments": ["\n".join(lines) + "\n"],
                "datasets": [{"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}],
                "result": {"path": "/output"},
                "resources": {"gpuCount": 1 if mode == "hf-matched" else 3, "sharedMemory": "100 GiB"},
                "context": {"priority": "urgent", "minRuntime": "1h", "autoResume": False},
                "constraints": {"cluster": ["ai2/holmes"]},
                "timeout": "90m",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--backend", choices=IMAGES, required=True)
    parser.add_argument("--campaign", default="update-zero-20260911-v1")
    parser.add_argument("--mode", choices=("original", "hf-matched"), default="original")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    document = specification(args.image, args.backend, campaign=args.campaign, mode=args.mode)
    if args.render_only:
        print(json.dumps(document, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="update-zero-launch-") as directory:
        path = Path(directory) / "experiment.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        subprocess.run(
            ["beaker", "experiment", "create", str(path), "--workspace", "ai2/open-instruct-dev", "--format", "json"],
            check=True,
        )


if __name__ == "__main__":
    main()
