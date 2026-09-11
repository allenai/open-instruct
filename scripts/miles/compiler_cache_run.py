"""Opt-in single-node compiler-cache wrapper; run before any training imports."""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import suppress
from importlib import metadata
from pathlib import Path

import tomllib

from open_instruct.miles import compiler_cache as cache


def command_version(command, environment):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False, env=environment)
        return {"returncode": result.returncode, "text": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def toolchain(environment):
    packages = {}
    for name in (
        "torch",
        "triton",
        "tilelang",
        "flash-linear-attention",
        "fla-core",
        "sglang",
        "flash-attn",
        "flash-attn-4",
        "transformer-engine",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    probe = command_version(
        [
            sys.executable,
            "-c",
            (
                "import json,torch; print(json.dumps({'torch_cuda':torch.version.cuda,"
                "'torch_git':torch.version.git_version,'gpus':["
                "{'name':p.name,'major':p.major,'minor':p.minor,'memory':p.total_memory,"
                "'sm_count':p.multi_processor_count} for p in "
                "[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]]}))"
            ),
        ],
        environment,
    )
    if probe.get("returncode") != 0:
        raise ValueError(f"Cannot fingerprint actual Torch/GPU runtime: {probe}")
    devices = json.loads(probe["text"].splitlines()[-1])
    if not devices["gpus"]:
        raise ValueError("Core compiler-cache qualification requires a visible GPU")
    driver = command_version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], environment)
    if driver.get("returncode") != 0 or not driver.get("text"):
        raise ValueError("Cannot fingerprint actual NVIDIA driver")
    return {
        "python": sys.version,
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "packages": packages,
        "torch_hardware": devices,
        "driver": driver,
        "nvcc": command_version([environment.get("CUDACXX", "nvcc"), "--version"], environment),
        "cc": command_version([environment.get("CC", "cc"), "--version"], environment),
        "cxx": command_version([environment.get("CXX", "c++"), "--version"], environment),
    }


def compiler_environment(environment):
    prefixes = (
        "TRITON_",
        "TORCHINDUCTOR_",
        "PYTORCH_",
        "OLMO_",
        "SGLANG_",
        "NVTE_",
        "FLA_",
        "TILELANG_",
        "FLASH_ATTENTION_",
        "CUTE_",
        "CUDNN_",
    )
    explicit = {
        "CC",
        "CXX",
        "CUDACXX",
        "TORCH_CUDA_ARCH_LIST",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
    }
    excluded = set(cache.FAMILIES.values()) | {"CUDA_VISIBLE_DEVICES"}
    return {
        key: cache.digest(value)
        for key, value in sorted(environment.items())
        if key not in excluded and (key.startswith(prefixes) or key in explicit)
    }


def validate_local_cache_controls(environment):
    for name in ("TRITON_CACHE_MANAGER", "TRITON_REMOTE_CACHE_BACKEND", "TRITON_OVERRIDE_DIR"):
        if environment.get(name):
            raise ValueError(f"Custom cache/override control is not qualified: {name}")
    for name, value in environment.items():
        if name.startswith("TORCHINDUCTOR_") and "REMOTE_CACHE" in name and value not in {"", "0"}:
            raise ValueError(f"Remote compiler cache is not qualified: {name}")


def input_identity(options, hardware, environment):
    source_roots = dict(value.split("=", 1) for value in options.source)
    if len(source_roots) != len(options.source):
        raise ValueError("Duplicate source names are ambiguous")
    return cache.fingerprint(
        image=options.image,
        runtime_lock=json.loads(options.runtime_lock.read_text()),
        sources={name: cache.source_identity(Path(path)) for name, path in source_roots.items()},
        model_config=json.loads(options.hf_config.read_text()),
        run_config=tomllib.loads(options.run_config.read_text()),
        toolchain=hardware,
        compiler_env=compiler_environment(environment),
    )


def run(options):
    if options.report.exists():
        raise ValueError("Use a new report path to preserve prior cache evidence")
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command:
        raise ValueError("Provide the bounded child command after --")
    options.local_parent.mkdir(parents=True, exist_ok=True)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": cache.SCHEMA,
        "mode": options.mode,
        "command": command,
        "started_unix": time.time(),
        "restore": [],
        "publish": [],
        "scope": "single-node process tree",
    }
    started = time.monotonic()
    local = Path(tempfile.mkdtemp(prefix="core-compiler-", dir=options.local_parent))
    cleanup = True
    try:
        environment = {**os.environ, **cache.local_environment(local)}
        validate_local_cache_controls(environment)
        report["private_local_root"] = str(local)
        report["cache_environment"] = {name: environment[name] for name in cache.FAMILIES.values()}
        probe_started = time.monotonic()
        # Hardware probe imports Torch in a subprocess after setting private caches.
        hardware = toolchain(environment)
        key, identity = input_identity(options, hardware, environment)
        report.update(fingerprint=key, identity=identity, probe_seconds=time.monotonic() - probe_started)
        for family in cache.FAMILIES:
            report["restore"].append(
                cache.restore(options.shared_root, local, key, family)
                if options.mode == "restore"
                else {"family": family, "status": "cold"}
            )
        command_started = time.monotonic()
        child = None
        try:
            child = subprocess.Popen(command, env=environment, start_new_session=True)
            code = child.wait()
        except BaseException:
            cleanup = False
            report["private_cache_retained"] = True
            if child is not None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise
        report.update(child_returncode=code, command_seconds=time.monotonic() - command_started)
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            survivors = False
        else:
            survivors = True
        report["surviving_child_group"] = survivors
        if survivors:
            cleanup = False
            report["private_cache_retained"] = True
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            raise RuntimeError("Child left its process group alive; terminated it and retained private caches")
        if code == 0 and options.publish:
            recheck_started = time.monotonic()
            after, _ = input_identity(options, hardware, environment)
            report["input_recheck_seconds"] = time.monotonic() - recheck_started
            if after != key:
                raise ValueError("Runtime source or configuration changed while the child ran; refusing publication")
            for family in cache.FAMILIES:
                try:
                    report["publish"].append(cache.publish(options.shared_root, local, key, family))
                except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
                    report["publish"].append({"family": family, "status": "rejected", "reason": str(error)})
        report["status"] = "completed" if code == 0 else "child_failed"
        return code
    except BaseException as error:
        report.update(status="wrapper_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if cleanup:
            shutil.rmtree(local)
        report["total_seconds"] = time.monotonic() - started
        with options.report.open("xb") as stream:
            stream.write(cache.encoded(report) + b"\n")


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cold", "restore"), required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--local-parent", type=Path, default=Path("/tmp"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="NAME=source tree; see docs")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
