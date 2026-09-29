"""Run the native synchronous Megatron GRPO driver in an owned local Ray cluster."""

import asyncio
import os
import runpy
from pathlib import Path

import ray
from miles.utils.arguments import parse_args
from miles.utils.tracking_utils.tracking import finish_tracking

import miles


def main():
    if miles.__file__ is None:
        raise RuntimeError("Native Miles source checkout is required")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("OI_GRPO_", "MILES_", "WANDB_", "OPEN_INSTRUCT_MATH_VERIFIER_"))
        or key in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDNN_HOME", "CUDNN_PATH")
    }
    ray.init(num_gpus=int(os.environ["OI_GRPO_RAY_GPUS"]), include_dashboard=False, runtime_env={"env_vars": env})
    try:
        args = parse_args()
        if args.use_opd or args.opd_kl_coef != 0 or args.fully_async or args.kl_coef != 0 or args.use_kl_loss:
            raise ValueError("Synchronous verifier GRPO cannot enable teacher, reference KL or async")
        namespace = runpy.run_path(str(Path(miles.__file__).resolve().parents[1] / "train.py"))
        asyncio.run(namespace["train"](args))
    finally:
        finish_tracking()
        ray.shutdown()


if __name__ == "__main__":
    main()
