"""Own a local Ray cluster and invoke the pinned native Miles training driver."""

import os
import runpy
from pathlib import Path

import ray

import miles


def main():
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("OI_OPD_", "MILES_", "WANDB_"))
        or key in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDNN_HOME", "CUDNN_PATH")
    }
    module_file = miles.__file__
    if module_file is None:
        raise RuntimeError("Native OPD requires the Miles source checkout")
    ray.init(num_gpus=3, include_dashboard=False, runtime_env={"env_vars": env})
    try:
        runpy.run_path(str(Path(module_file).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
