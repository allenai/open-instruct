"""Run the native Megatron GRPO driver (synchronous or fully async) in an owned local Ray cluster."""

import asyncio
import os
import runpy
from pathlib import Path

import ray
from miles.utils.arguments import parse_args
from miles.utils.tracking_utils.tracking import finish_tracking

import miles
from open_instruct.miles import megatron_grpo_config


def main():
    if miles.__file__ is None:
        raise RuntimeError("Native Miles source checkout is required")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("OI_GRPO_", "OI_DPPO_", "MILES_", "WANDB_", "OPEN_INSTRUCT_MATH_VERIFIER_"))
        or key in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDNN_HOME", "CUDNN_PATH")
    }
    ray.init(num_gpus=int(os.environ["OI_GRPO_RAY_GPUS"]), include_dashboard=False, runtime_env={"env_vars": env})
    try:
        args = parse_args()
        if args.use_opd or args.opd_kl_coef != 0 or args.kl_coef != 0 or args.use_kl_loss:
            raise ValueError("Verifier GRPO cannot enable a teacher or reference KL")
        driver = "train.py"
        if args.fully_async:
            if not args.use_rollout_logprobs:
                raise ValueError("Async verifier GRPO requires --use-rollout-logprobs")
            driver = "train_async.py"
            # Native parsing selects its own producer; install the managed one afterwards and keep
            # evaluation on that same instance (as opd_train does for async OPD).
            args.rollout_function_path = megatron_grpo_config.ASYNC_ROLLOUT
            args.eval_function_path = megatron_grpo_config.ASYNC_ROLLOUT
        namespace = runpy.run_path(str(Path(miles.__file__).resolve().parents[1] / driver))
        asyncio.run(namespace["train"](args))
    finally:
        finish_tracking()
        ray.shutdown()


if __name__ == "__main__":
    main()
