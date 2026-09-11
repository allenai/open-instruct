"""Profile the unmodified Core100 scorer on one retained batch, without serving."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest import mock

import torch
from miles.backends.training_utils import parallel
from miles.ray.rollout import train_data_conversion
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample
from scripts.miles import gsm8k_parity
from torch import distributed as dist
from triton.runtime import cache, jit

from open_instruct.miles import actor, data, models


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_scores(reference, current):
    if len(reference) != len(current) or any(a.shape != b.shape for a, b in zip(reference, current, strict=True)):
        return {"valid": False, "reason": "score shapes differ"}
    differences = torch.cat(
        [(a.float() - b.float()).abs().reshape(-1) for a, b in zip(reference, current, strict=True)]
    )
    finite = bool(torch.isfinite(differences).all())
    return {
        "valid": finite and bool((differences == 0).all()),
        "finite": finite,
        "tokens": differences.numel(),
        "max_abs": float(differences.max()) if finite else None,
        "mean_abs": float(differences.mean()) if finite else None,
    }


class CompilerObservation:
    """JIT in-memory misses include disk hits; artifact writes are separate evidence."""

    def __init__(self):
        self.misses = []
        self.writes = Counter()
        self.original_compile = jit.JITFunction._do_compile
        self.original_put = cache.FileCacheManager.put

    def compile(self, function, key, signature, device, constexprs, options, attrs, warmup):
        started = time.perf_counter()
        try:
            return self.original_compile(function, key, signature, device, constexprs, options, attrs, warmup)
        finally:
            self.misses.append(
                {
                    "kernel": function.fn.__module__ + "." + function.fn.__name__,
                    "key_sha256": hashlib.sha256(str(key).encode()).hexdigest(),
                    "constexprs": repr(constexprs),
                    "wall_seconds": time.perf_counter() - started,
                }
            )

    def put(self, manager, value, filename, binary=True):
        result = self.original_put(manager, value, filename, binary=binary)
        self.writes[Path(filename).suffix] += 1
        return result

    def summary(self):
        return {
            "jit_in_memory_misses": self.misses,
            "jit_miss_count": len(self.misses),
            "jit_miss_wall_seconds": sum(row["wall_seconds"] for row in self.misses),
            "cache_artifact_writes_by_extension": dict(self.writes),
            "scope": "_score only; misses include disk-cache loading; writes do not count unique kernels",
        }


def scored_pass(worker, batches, name):
    observation = CompilerObservation()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    start.record()
    with (
        mock.patch.object(jit.JITFunction, "_do_compile", lambda *args: observation.compile(*args)),
        mock.patch.object(cache.FileCacheManager, "put", lambda *args, **kwargs: observation.put(*args, **kwargs)),
    ):
        scores = worker._score(worker.train_module, batches, use_replay=False)
    end.record()
    torch.cuda.synchronize()
    row = {
        "name": name,
        "wall_seconds": time.perf_counter() - started,
        "cuda_stream_elapsed_seconds": start.elapsed_time(end) / 1000,
        "cuda_event_scope": "stream elapsed, including idle time; not summed GPU busy time",
        **observation.summary(),
    }
    return [value.detach().cpu().clone() for value in scores], row


def run(root, output, manifest, rollout, repeats):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("Frozen scorer qualification requires the original two-rank EP2 topology")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "world_size": world, "passes": [], "valid": False}
    destination = output / f"rank{rank}.json"
    try:
        if file_digest(__file__) != manifest["worker_sha256"]:
            raise ValueError("Embedded diagnostic worker digest differs")
        observed = {}
        for module, expected in manifest["module_sha256"].items():
            path = inspect.getfile(importlib.import_module(module))
            observed[module] = {"path": path, "sha256": file_digest(path)}
            if observed[module]["sha256"] != expected:
                raise ValueError(f"Frozen source differs: {module}")
        report["sources"] = observed
        report["manifest"] = manifest
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
        # This module and its configuration are checked against the original image source.
        configuration = gsm8k_parity.configuration(root)
        sys.argv = ["frozen-core-score-profile", *configuration.arguments()]
        args = arguments.parse_args()
        gsm8k_parity.effective_settings(args)
        torch.manual_seed(args.seed)
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        initialized = time.perf_counter()
        worker.args = args
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model = worker.train_module.model
        torch.cuda.synchronize()
        report["model_initialization_seconds"] = time.perf_counter() - initialized
        path = root / "core" / "rollouts" / f"{rollout}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = [Sample.from_dict(sample) for sample in payload["samples"]]
        train = train_data_conversion.convert_samples_to_train_data(args, samples, payload["metadata"], None, None)
        shards = train_data_conversion.split_train_data_by_dp_raw(args, train, dp_size=world)
        partition = list(shards[rank]["partition"])
        shard = train_data_conversion.process_rollout_data_shard(args, shards[rank])
        for key, dtype in (("tokens", torch.long), ("loss_masks", torch.int)):
            shard[key] = [torch.tensor(value, device="cuda", dtype=dtype) for value in shard[key]]
        batches = data.sample_batches(shard, args.olmo_core.max_sequence_length)
        report["input"] = {
            "path": str(path),
            "sha256": file_digest(path),
            "rollout": rollout,
            "partition": partition,
            "lengths": shard["total_lengths"],
            "response_lengths": shard["response_lengths"],
            "balance_data": args.balance_data,
        }
        report["checkpoint_scope"] = (
            "Original initial HF weights; retained rollout supplies tokens, not trained weights"
        )
        report["ingress_scope"] = (
            "Filesystem load/DP partition occurs before scoring timers; no Ray object-store fetch"
        )
        reference = None
        for index in range(repeats + 1):
            scores, row = scored_pass(worker, batches, "cold" if index == 0 else f"warm{index}")
            row["residual"] = compare_scores(scores if reference is None else reference, scores)
            if reference is None:
                reference = scores
            report["passes"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
        # This pass is diagnostic instrumentation and is excluded from regular warm means.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        ) as profile:
            original_forward = worker._forward

            def forward(*args, **kwargs):
                with torch.profiler.record_function("core_score_forward"):
                    return original_forward(*args, **kwargs)

            original_logprobs = actor.miles_loss.get_log_probs_and_entropy

            def logprobs(*args, **kwargs):
                with torch.profiler.record_function("core_score_logprobs"):
                    return original_logprobs(*args, **kwargs)

            with (
                mock.patch.object(worker, "_forward", forward),
                mock.patch.object(actor.miles_loss, "get_log_probs_and_entropy", logprobs),
            ):
                scores, row = scored_pass(worker, batches, "profile_diagnostic_only")
        row["residual"] = compare_scores(reference, scores)
        report["passes"].append(row)
        profile.export_chrome_trace(str(output / f"rank{rank}-trace.json"))
        report["valid"] = all(row["residual"]["valid"] for row in report["passes"])
        rejected = torch.tensor(int(not report["valid"]), device="cuda")
        dist.all_reduce(rejected, op=dist.ReduceOp.MAX)
        if rejected.item():
            raise ValueError("Repeated scoring changed or was nonfinite; timings and residuals retained")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rollout", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    if options.repeats < 1 or options.rollout < 0:
        parser.error("Need a nonnegative rollout and at least one warm repeat")
    run(options.root, options.output, json.loads(options.manifest.read_text()), options.rollout, options.repeats)
