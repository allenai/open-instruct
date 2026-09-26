"""Runtime identity and environment checks for worker compiler caches."""

import json
import platform
import subprocess
import sys
from importlib import metadata

from open_instruct.miles.infrastructure import compiler_cache as cache


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
    excluded = set(cache.FAMILIES.values()) | {
        "CUDA_VISIBLE_DEVICES",
        # Locations, not compiler options. MILES can choose a fresh directory
        # per serving process; including it prevents an identical restart hit.
        "SGLANG_DG_CACHE_DIR",
        "TILELANG_TMP_DIR",
    }
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
