"""Own a local Ray cluster and invoke the pinned native Miles training driver."""

import asyncio
import os
import runpy
from pathlib import Path

import ray
from miles.utils.arguments import parse_args
from miles.utils.tracking_utils.tracking import finish_tracking

import miles
from open_instruct.miles import opd_config


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
    ray.init(
        num_gpus=int(os.environ.get("OI_OPD_RAY_GPUS", "3")), include_dashboard=False, runtime_env={"env_vars": env}
    )
    try:
        args = parse_args()
        driver = "train.py"
        if args.fully_async:
            driver = "train_async.py"
            # Native parsing selects its own producer. Install the OPD adapter only
            # afterwards, keeping shared-engine evaluation on that same instance.
            args.rollout_function_path = opd_config.ASYNC_ROLLOUT
            args.eval_function_path = opd_config.ASYNC_ROLLOUT
        namespace = runpy.run_path(str(Path(module_file).resolve().parents[1] / driver))
        asyncio.run(namespace["train"](args))
    finally:
        finish_tracking()
        ray.shutdown()


if __name__ == "__main__":
    main()
