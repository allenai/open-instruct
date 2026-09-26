"""Replay identical retained batches through native Core score/backward/optimizer."""

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from miles.backends.training_utils import parallel
from miles.ray.rollout import train_data_conversion
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample
from scripts.miles import trainer_capacity_config
from scripts.miles.profile_frozen_core_scores import CompilerObservation
from torch import distributed as dist
from torch._dynamo import utils as dynamo_utils
from triton.runtime import cache, jit

from open_instruct.miles.publication.state import PolicyClock
from open_instruct.miles.training import actor, models, scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(trainer_capacity_config.VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    output = parsed.output
    output.mkdir(parents=True, exist_ok=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("This matched screen requires EP2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    group = GroupInfo(rank=rank, size=world, group=dist.group.WORLD, gloo_group=gloo)
    trivial = GroupInfo(rank=0, size=1, group=None)
    parallel.set_parallel_state(
        parallel.ParallelState(
            intra_dp=group,
            intra_dp_cp=group,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )
    run = trainer_capacity_config.configuration(parsed.variant, str(output))
    sys.argv = ["trainer-capacity", *run.arguments()]
    args = arguments.parse_args()
    args.use_wandb = False
    torch.manual_seed(args.seed)
    report = dict(
        variant=parsed.variant,
        rank=rank,
        passed=False,
        batches=[],
        recipe=run.arguments(),
        kernel_environment={key: os.environ.get(key, "0") for key in trainer_capacity_config.KERNEL_FLAGS},
        scope="Fixed retained batches and initial HF policy. No fresh sampling, reward call, Ray ingress or weight delivery. Publication clock advances logically after each optimizer update.",
    )
    destination = output / f"trainer-capacity-rank{rank}.json"

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        started = time.perf_counter()
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        torch.cuda.synchronize()
        report["initialization_seconds"] = time.perf_counter() - started
        observation = CompilerObservation()
        phases = {}
        current_shard = [None]

        def timed(name, function):
            def call(*a, **kw):
                torch.cuda.synchronize()
                before = time.perf_counter()
                try:
                    return function(*a, **kw)
                finally:
                    torch.cuda.synchronize()
                    phases[name] = phases.get(name, 0) + time.perf_counter() - before

            return call

        with contextlib.ExitStack() as stack:
            for owner, name, label in (
                (worker, "_score", "scoring"),
                (worker.train_module, "train_batch_with_loss", "forward_loss_backward"),
                (worker.train_module, "optim_step", "optimizer"),
            ):
                stack.enter_context(mock.patch.object(owner, name, timed(label, getattr(owner, name))))
            stack.enter_context(mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo))
            stack.enter_context(
                mock.patch.object(
                    actor.miles_data,
                    "process_rollout_data",
                    side_effect=lambda *a, **kw: (current_shard[0], contextlib.nullcontext()),
                )
            )
            stack.enter_context(mock.patch.object(jit.JITFunction, "_do_compile", lambda *a: observation.compile(*a)))
            stack.enter_context(
                mock.patch.object(cache.FileCacheManager, "put", lambda *a, **kw: observation.put(*a, **kw))
            )
            for update in range(16):
                source = Path(trainer_capacity_config.SOURCE) / "rollouts" / f"{update}.pt"
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                payload = torch.load(source, map_location="cpu", weights_only=False)
                samples = [Sample.from_dict(sample) for sample in payload["samples"]]
                train = train_data_conversion.convert_samples_to_train_data(
                    args, samples, payload["metadata"], None, None
                )
                shards = train_data_conversion.split_train_data_by_dp_raw(args, train, dp_size=world)
                shard = train_data_conversion.process_rollout_data_shard(args, shards[rank])
                current_shard[0] = shard
                observation.misses.clear()
                observation.writes.clear()
                phases.clear()
                torch.cuda.synchronize()
                before = time.perf_counter()
                worker.train(update, None)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                worker.clock.published()
                record = dict(
                    update=update,
                    source=str(source),
                    source_sha256=digest,
                    samples=len(samples),
                    local_tokens=sum(t.numel() for t in shard["tokens"]),
                    local_response_tokens=sum(shard["response_lengths"]),
                    seconds=elapsed,
                    phases=dict(phases),
                    other_seconds=elapsed - sum(phases.values()),
                    memory_allocated_peak=torch.cuda.max_memory_allocated(),
                    memory_reserved_peak=torch.cuda.max_memory_reserved(),
                    compilation=copy.deepcopy(observation.summary()),
                    dynamo_cumulative={k: sum(v.values()) for k, v in dynamo_utils.counters.items()},
                    dynamo_stats=dict(dynamo_utils.counters["stats"]),
                )
                record["compilation"]["scope"] = (
                    "Whole trainer call; misses include disk hits. Separate artifact writes and kernel identities distinguish cache reuse from compilation."
                )
                report["batches"].append(record)
                save()
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            variant=parsed.variant,
                            update=update,
                            seconds=elapsed,
                            phases=dict(phases),
                            jit_misses=record["compilation"]["jit_miss_count"],
                        )
                    ),
                    flush=True,
                )
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if "observation" in locals():
            report["failure_compilation"] = copy.deepcopy(observation.summary())
            report["failure_phases"] = dict(phases)
            report["failure_update"] = update if "update" in locals() else None
        raise
    finally:
        save()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
