"""Run the native Megatron GRPO driver (synchronous or fully async) in an owned local Ray cluster."""

import asyncio
import os
import runpy
from pathlib import Path

import ray
from miles.ray.placement_group import create_rollout_components, create_training_models, update_weights
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

import miles
from open_instruct.miles.megatron_grpo import grpo_config


async def evaluate(args):
    """Run the native async driver's pre-training eval of the starting model, then release everything.

    Mirrors the setup of the pinned ``train_async.train`` up to its rollout-0 eval, so scores match
    the in-training eval. The native driver has no async eval-only mode: it starts generating the
    first training batch even when there is nothing to train.
    """
    configure_logger(args, source=MainProcessIdentity())
    # Hold the handle as train_async does: the named worker manager is not detached, so Ray kills it once the
    # last handle is dropped, and the rollout servers then fail to look it up.
    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    inference_controller, rollout_executor, _ = await create_rollout_components(args)
    actor_model, critic_model = await create_training_models(args, inference_controller, rollout_executor)
    await update_weights(actor_model, rollout_executor)
    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    await inference_controller.prepare_eval()
    await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint, force=True)
    await eval_dispatcher.drain()
    await rollout_executor.dispose.remote()
    await inference_controller.dispose()
    await actor_model.dispose()
    if critic_model is not None:
        await critic_model.dispose()


def main():
    if miles.__file__ is None:
        raise RuntimeError("Native Miles source checkout is required")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("OI_GRPO_", "OI_DPPO_", "OI_MILES_", "MILES_", "WANDB_", "OPEN_INSTRUCT_MATH_VERIFIER_"))
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
            args.rollout_function_path = grpo_config.ASYNC_ROLLOUT
            args.eval_function_path = grpo_config.ASYNC_ROLLOUT
        if os.environ.get(grpo_config.EVAL_ONLY_ENV) == "1":
            if not args.fully_async:
                raise ValueError("Eval-only verifier GRPO requires the fully async producer")
            asyncio.run(evaluate(args))
            return
        namespace = runpy.run_path(str(Path(miles.__file__).resolve().parents[1] / driver))
        asyncio.run(namespace["train"](args))
    finally:
        finish_tracking()
        ray.shutdown()


if __name__ == "__main__":
    main()
