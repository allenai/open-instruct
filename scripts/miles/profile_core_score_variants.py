"""Compare successive retained batches under isolated Core kernel variants."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch
from miles.backends.training_utils import parallel
from miles.ray.rollout import train_data_conversion
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample
from olmo_core.kernels import swiglu
from scripts.miles import gsm8k_parity
from scripts.miles.profile_frozen_core_scores import compare_scores, file_digest, scored_pass
from torch import distributed as dist

from open_instruct.miles import actor, data, models


def sources():
    kernel = Path(inspect.getfile(swiglu))
    root = kernel.parents[2]
    files = {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py")) if path != kernel}
    modules = (actor, data, models, gsm8k_parity)
    return {
        "kernel_sha256": file_digest(kernel),
        "other_core_python_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "other_core_python_files": len(files),
        "oi_modules": {module.__name__: file_digest(inspect.getfile(module)) for module in modules},
    }


def compare_runs(root, manifest):
    report = {"valid": True, "comparisons": [], "optimizer_updates": 0}
    for rank in (0, 1):
        left = json.loads((root / "parent" / f"rank{rank}.json").read_text())
        right = json.loads((root / "candidate" / f"rank{rank}.json").read_text())
        if not left["valid"] or not right["valid"]:
            raise ValueError("An arm failed its internal repeated-score gate")
        if left["sources"]["other_core_python_sha256"] != right["sources"]["other_core_python_sha256"]:
            raise ValueError("Core Python sources other than SwiGLU differ")
        if left["sources"]["oi_modules"] != right["sources"]["oi_modules"]:
            raise ValueError("Open-instruct runtime sources differ")
        if left["recipe_argv"] != right["recipe_argv"]:
            raise ValueError("Effective recipe arguments differ")
        if left["inputs"] != right["inputs"]:
            raise ValueError("Retained batch inputs or partitions differ")
        for arm, document in (("parent", left), ("candidate", right)):
            if document["sources"]["kernel_sha256"] != manifest[f"{arm}_sha256"]:
                raise ValueError(f"{arm} kernel source differs")
        for rollout in range(5, 10):
            values = [
                torch.load(root / arm / f"scores-{rollout}-{rank}.pt", weights_only=True)
                for arm in ("parent", "candidate")
            ]
            residual = compare_scores(*values)
            report["comparisons"].append({"rank": rank, "rollout": rollout, **residual})
            report["valid"] = report["valid"] and residual["valid"]
    return report


def run(root, output, arm, manifest):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise ValueError("Qualification requires EP2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": rank,
        "arm": arm,
        "world_size": world,
        "passes": [],
        "inputs": [],
        "valid": False,
        "optimizer_updates": 0,
    }
    destination = output / f"rank{rank}.json"
    try:
        report["sources"] = sources()
        if report["sources"]["kernel_sha256"] != manifest[f"{arm}_sha256"]:
            raise ValueError("Installed variant kernel source differs")
        if file_digest(__file__) != manifest["worker_sha256"]:
            raise ValueError("Diagnostic worker source differs")
        report["manifest"] = manifest
        dist.init_process_group("nccl")
        group = GroupInfo(rank=rank, size=world, group=dist.group.WORLD, gloo_group=dist.new_group(backend="gloo"))
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
        configuration = gsm8k_parity.configuration(root)
        report["recipe_argv"] = configuration.arguments()
        sys.argv = ["core-score-variants", *report["recipe_argv"]]
        args = arguments.parse_args()
        gsm8k_parity.effective_settings(args)
        torch.manual_seed(args.seed)
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        started = time.perf_counter()
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model = worker.train_module.model
        torch.cuda.synchronize()
        report["model_initialization_seconds"] = time.perf_counter() - started
        for rollout in range(5, 10):
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
            report["inputs"].append(
                {
                    "path": str(path),
                    "sha256": file_digest(path),
                    "rollout": rollout,
                    "partition": partition,
                    "lengths": shard["total_lengths"],
                    "response_lengths": shard["response_lengths"],
                }
            )
            first, row = scored_pass(worker, batches, f"batch{rollout}-first")
            row["residual"] = compare_scores(first, first)
            report["passes"].append(row)
            torch.save(first, output / f"scores-{rollout}-{rank}.pt")
            repeated, row = scored_pass(worker, batches, f"batch{rollout}-repeat")
            row["residual"] = compare_scores(first, repeated)
            report["passes"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
        report["valid"] = all(row["residual"]["valid"] for row in report["passes"])
        rejected = torch.tensor(int(not report["valid"]), device="cuda")
        dist.all_reduce(rejected, op=dist.ReduceOp.MAX)
        if rejected.item():
            raise ValueError("Repeated scores differ; full timings and residuals retained")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--arm", choices=("parent", "candidate"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.arm:
        run(args.root, args.output / args.arm, args.arm, manifest)
    else:
        result = compare_runs(args.output, manifest)
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["valid"]:
            raise ValueError("Cross-arm exact score gate failed; residuals retained")


if __name__ == "__main__":
    main()
