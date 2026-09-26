"""Small actual-RL refresh qualification with retained dumps and fresh-process resume."""

import argparse
import asyncio
import dataclasses
import json
import os
import time
from pathlib import Path

import ray
from scripts.miles import gsm8k_parity

from open_instruct.miles.configuration.config import RunConfig
from open_instruct.miles.execution.driver import train
from open_instruct.miles.execution.workflow import parse_runtime


def configuration(campaign, output, *, mode="refresh", updates=4, resume=False):
    original = gsm8k_parity.configuration(campaign, updates=updates, eval_interval=2, save_interval=2)
    core = dataclasses.replace(
        original.core,
        publication_mode=mode,
        max_policy_lag=2,
        engine_drain_timeout=900,
        row_specialization="dynamic",
        max_train_rollout_logprob_abs_diff=None,
        replay_diagnostics=True,
        scoring_pass_required=True,
        scoring_check_interval=1,
    )
    options = dict(original.miles)
    for key in list(options):
        if key.startswith("wandb_"):
            del options[key]
    options.update(
        num_gpus_per_node=4,
        rollout_num_gpus=2,
        fully_async=True,
        use_miles_router=True,
        use_rollout_routing_replay=True,
        use_tis=True,
        use_rollout_logprobs=False,
        tis_clip=2.0,
        tis_clip_low=0.5,
        async_max_concurrent_samples=32,
        async_data_buffer_capacity_factor=1.0,
        async_unused_samples_handler="retry",
        rollout_submission_granularity="group",
        sglang_cuda_graph_backend_decode="disabled",
        sglang_cuda_graph_backend_prefill="disabled",
        sglang_disable_radix_cache=False,
        sglang_max_running_requests=8,
        sglang_server_concurrency=8,
        sglang_max_mamba_cache_size=16,
        sglang_chunked_prefill_size=8192,
        eval_prompt_data=["gsm8k", str(output / "eval.jsonl")],
        save=str(output / "metrics"),
        save_debug_rollout_data=str(output / "rollouts/{rollout_id}.pt"),
        rollout_sample_rate=1.0,  # This diagnostic audits complete saved batches.
        use_wandb=False,
        skip_eval_before_train=resume,
    )
    if resume:
        options["load"] = str(output / "metrics")
    return RunConfig(core, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("refresh", "barrier"), default="refresh")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    opt = parser.parse_args()
    opt.output.mkdir(parents=True, exist_ok=opt.resume)
    if not opt.resume:
        rows = (opt.campaign / "eval.jsonl").read_text().splitlines()[:8]
        (opt.output / "eval.jsonl").write_text("\n".join(rows) + "\n")
    config = configuration(opt.campaign, opt.output, mode=opt.mode, updates=opt.updates, resume=opt.resume)
    invocation = "resume" if opt.resume else "initial"
    (opt.output / f"{invocation}-arguments.json").write_text(json.dumps(config.arguments(), indent=2) + "\n")
    args = parse_runtime(config)
    expected = "RefreshingRolloutFn" if opt.mode == "refresh" else "ManagedFullyAsyncRolloutFn"
    if not args.rollout_function_path.endswith(expected):
        raise RuntimeError("Runtime did not select the requested rollout producer")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    started = time.monotonic()
    ray.init(num_gpus=4, num_cpus=24, include_dashboard=False, object_store_memory=2 * 1024**3)
    try:
        result = asyncio.run(train(args))
        expected_ids = list(range(args.start_rollout_id, args.num_rollout))
        if result["completed_rollout_ids"] != expected_ids or (opt.resume and args.start_rollout_id != 4):
            raise RuntimeError(f"Unexpected optimizer/resume boundaries: {result}")
        (opt.output / f"{invocation}-result.json").write_text(
            json.dumps(
                {
                    **result,
                    "wall_seconds": time.monotonic() - started,
                    "scope": "Small integration/performance qualification, not a learning-quality comparison",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
