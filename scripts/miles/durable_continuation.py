"""Fixed-batch Core continuation across fresh processes, with bounded state hashing.

The real actor, native optimizer/checkpoint and MILES dataset cursor are exercised.
Responses are deterministic fixtures scored by Core; this is not a serving test.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss_hub import losses
from miles.rollout.data_source import RolloutDataSource
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from scripts.miles.checkpoint_weights import SafeTensorState
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import distributed as dist
from torch.distributed.tensor import DTensor
from transformers import PreTrainedTokenizerFast

from open_instruct.miles import actor, checkpoint, models, scheduler
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock, atomic_json

HORIZON = 4
BOUNDARY = 2
CHUNK_BYTES = 16 * 1024**2


def runtime_lock():
    image_path = Path("/opt/core-rl/build/runtime/miles/runtime.lock.json")
    path = (
        image_path if image_path.is_file() else Path(__file__).resolve().parents[2] / "runtime/miles/runtime.lock.json"
    )
    return json.loads(path.read_text())


def tensor_record(value):
    """Hash local bytes in bounded CPU transfers; never gather global EP/DP tensors."""
    local = value.to_local() if isinstance(value, DTensor) else value
    flat = local.detach().reshape(-1)
    digest = hashlib.sha256()
    width = max(1, CHUNK_BYTES // flat.element_size())
    for offset in range(0, flat.numel(), width):
        chunk = flat[offset : offset + width].contiguous().cpu()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return {
        "sha256": digest.hexdigest(),
        "dtype": str(local.dtype),
        "local_shape": list(local.shape),
        "global_shape": list(value.shape),
        "placements": [str(p) for p in value.placements] if isinstance(value, DTensor) else [],
        "bytes": local.numel() * local.element_size(),
    }


def cursor_state(source):
    return {
        name: getattr(source, name)
        for name in ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata")
    }


def capture(worker, source):
    # OLMoDDPOptimizer.state_dict() mutates/frees live storage. Inspect states directly.
    model = {name: tensor_record(value) for name, value in sorted(worker.model.state_dict().items())}
    optim = {name: tensor_record(value) for name, value in sorted(worker.optimizer.states.items())}
    names = {name for group in worker.optimizer.param_groups for name in group["named_params"]}
    for name in names:
        for suffix in ("main", "exp_avg", "exp_avg_sq", "step"):
            if f"{name}.{suffix}" not in optim:
                raise ValueError(f"Missing optimizer state: {name}.{suffix}")
    np_state = np.random.get_state()
    return {
        "model": model,
        "optimizer": optim,
        "scheduler": worker.lr_scheduler.state_dict(),
        "clock": worker.clock.as_dict(),
        "trainer_global_step": worker.train_module._trainer.global_step,
        "cursor": cursor_state(source),
        "rng": {
            "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "numpy": hashlib.sha256(np_state[1].tobytes() + repr((np_state[0], *np_state[2:])).encode()).hexdigest(),
            "torch": tensor_record(torch.get_rng_state())["sha256"],
            "cuda": tensor_record(torch.cuda.get_rng_state())["sha256"],
        },
    }


def compare_states(expected, actual):
    failures = []
    for category in ("model", "optimizer"):
        left, right = expected[category], actual[category]
        if set(left) != set(right):
            failures.append(f"{category}: state keys differ")
        failures.extend(f"{category}/{name}" for name in left.keys() & right.keys() if left[name] != right[name])
    for category in ("scheduler", "clock", "trainer_global_step", "cursor", "rng"):
        if expected[category] != actual[category]:
            failures.append(category)
    return failures


def prepare(root, hf):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "inputs.jsonl").exists():
        raise FileExistsError("Use a fresh output root to preserve prior continuation evidence")
    if hf is None:
        subprocess.run([sys.executable, "tests/miles/ep_contract.py", "bootstrap", str(root / "fixture")], check=True)
        hf = root / "fixture/hf"
        tokenizer = Tokenizer(WordLevel({str(i): i for i in range(256)}, unk_token="0"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="0", eos_token="1", pad_token="2"
        ).save_pretrained(hf)
    hf = hf.resolve()
    headers, weight_bytes = {}, 0
    for shard in sorted(hf.glob("*.safetensors")):
        with shard.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            raw = stream.read(length)
        header = json.loads(raw)
        weight_bytes += sum(
            value["data_offsets"][1] - value["data_offsets"][0]
            for key, value in header.items()
            if key != "__metadata__"
        )
        headers[shard.name] = hashlib.sha256(raw).hexdigest()
    if not headers:
        raise ValueError("HF source has no safetensors shards")
    # Native checkpoint persists FP32 master + both moments and small metadata.
    # This conservative bound also budgets model/persistent-buffer representations.
    estimate = weight_bytes * 8 + 2 * 1024**3
    available = shutil.disk_usage(root).free
    if available < estimate:
        raise ValueError(f"Insufficient checkpoint disk: free={available}, conservative_required={estimate}")
    rows = [
        {"input": f"Controlled fixed batch {i}", "label": "fixture", "metadata": {"fixture_index": i}}
        for i in range(HORIZON)
    ]
    (root / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_json(
        root / "preparation.json",
        {
            "hf": str(hf),
            "config_sha256": hashlib.sha256((hf / "config.json").read_bytes()).hexdigest(),
            "header_sha256": headers,
            "hf_weight_bytes": weight_bytes,
            "conservative_checkpoint_disk_bytes": estimate,
            "available_disk_bytes": available,
            "fixed_horizon": HORIZON,
            "checkpoint_boundary": BOUNDARY,
            "input_sha256": hashlib.sha256((root / "inputs.jsonl").read_bytes()).hexdigest(),
            "runtime_lock": runtime_lock(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits": [
                "Synthetic fixed responses/rewards, no SGLang or weight transport",
                "Identical trainer topology only",
                "Runtime differs from the frozen Core100 GSM8K image",
            ],
        },
    )


def fixed_rollout(groups, rank, world, step, vocab):
    samples = groups[0]
    if len(groups) != 1 or len(samples) != 4 or any(s.metadata["fixture_index"] != step for s in samples):
        raise ValueError("MILES cursor selected a different fixed batch")
    result = {
        key: [] for key in ("tokens", "total_lengths", "response_lengths", "loss_masks", "rewards", "weight_versions")
    }
    for index in range(rank, 4, world):
        sample = samples[index]
        if sample.group_index != step or sample.index != step * 4 + index:
            raise ValueError("MILES sample/group cursor differs from the fixed continuation sequence")
        total, response = 16 + index * 8, 8 + index * 4
        result["tokens"].append((torch.arange(total, device="cuda") + 17 * step + 11 * index + 3) % vocab)
        result["total_lengths"].append(total)
        result["response_lengths"].append(response)
        mask = torch.ones(response, device="cuda")
        mask[1] = 0
        result["loss_masks"].append(mask)
        result["rewards"].append([1.0, 0.0, 1.0, 0.0][index])
        result["weight_versions"].append([str(step)])
    return result


def run(root, phase, backend, *, checkpoint_options=None, continue_after_save=False, verify_export=False):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (1, 2):
        raise ValueError("Continuation qualification supports local EP1 or full-model EP2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    started = time.monotonic()
    try:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
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
        prepared = json.loads((root / "preparation.json").read_text())
        if hashlib.sha256((root / "inputs.jsonl").read_bytes()).hexdigest() != prepared["input_sha256"]:
            raise ValueError("Fixed inputs changed")
        output = root / phase
        config = RunConfig(
            CoreConfig(
                expert_parallel_size=world,
                attention_backend=backend,
                max_sequence_length=128,
                activation_checkpointing=True,
                diagnostic_interval=0,
                max_train_rollout_logprob_abs_diff=0.0,
            ),
            dict(
                hf_checkpoint=prepared["hf"],
                global_batch_size=4,
                rollout_batch_size=1,
                n_samples_per_prompt=4,
                num_rollout=HORIZON,
                actor_num_gpus_per_node=world,
                debug_train_only=True,
                save=str(output),
                rollout_global_dataset=True,
                prompt_data=str(root / "inputs.jsonl"),
                input_key="input",
                label_key="label",
                metadata_key="metadata",
                lr=1e-6,
                min_lr=1e-7,
                lr_decay_style="linear",
                lr_decay_iters=HORIZON,
                lr_warmup_iters=0,
                weight_decay=0.0,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                clip_grad=1.0,
                seed=17,
                rollout_seed=17,
                disable_grpo_std_normalization=True,
            ),
        )
        if phase == "resumed":
            config.miles["load"] = str(root / "split")
        sys.argv = ["durable-continuation", *config.arguments()]
        args = arguments.parse_args()
        if args.lr_decay_iters != HORIZON or args.num_rollout != HORIZON or args.rollout_shuffle:
            raise ValueError("The four-step scheduler horizon and fixed input order must match every process")
        source = RolloutDataSource(args)
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
        worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker.clock = PolicyClock()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        snapshots, score_checks, score_hashes, exports, save_timings = {}, [], {}, {}, []
        if checkpoint_options is not None:
            original_save = worker.train_module.save_state_dict_direct

            def measured_save(path, **kwargs):
                timings = original_save(path, **kwargs, **checkpoint_options)
                save_timings.append(timings)
                atomic_json(output / f"save-rank{rank}.json", timings)
                print("CHECKPOINT_PROFILE", json.dumps({"rank": rank, **timings}), flush=True)
                return timings

            worker.train_module.save_state_dict_direct = measured_save

        def export_at_boundary():
            export_path = output / "hf-boundary"
            worker.export_hf(BOUNDARY - 1, str(export_path))
            # Read the actual written artifact and compare the complete exported inventory.
            # Every rank participates in the native EP gather; only rank 0 reads the file.
            expected = {
                name: tensor_record(value)
                for name, value in models.iter_export_state(worker.train_module, worker.hf_config)
            }
            if rank == 0:
                with SafeTensorState(export_path) as stored:
                    actual = {name: tensor_record(value) for name, value in stored.items()}
                if actual != expected:
                    raise ValueError("HF export file differs from live model export")
            dist.barrier()
            return expected

        with mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo):
            if phase == "resumed":
                checkpoint.restore(worker)
                worker.train_module._trainer.global_step = worker.clock.completed_steps
                source.load(BOUNDARY - 1)
                snapshots["restored2"] = capture(worker, source)
                if verify_export:
                    exports["boundary"] = export_at_boundary()
                if worker.clock.completed_steps != BOUNDARY or source.sample_offset != BOUNDARY:
                    raise ValueError("Resume did not restore the two-update boundary")
            worker.clock.published()  # No serving process in this trainer-only qualification.
            stop = BOUNDARY if phase == "split" and not continue_after_save else HORIZON
            for step in range(worker.clock.next_rollout_id, stop):
                rollout = fixed_rollout(source.get_samples(1), rank, world, step, worker.hf_config.vocab_size)
                original_score = worker._score
                original_loss = actor.miles_loss.loss_function
                original_log_probs = losses.get_log_probs_and_entropy
                observed = torch.zeros(3, device="cuda", dtype=torch.float64)

                def scored(module, batches, *, use_replay, original_score=original_score, rollout=rollout, step=step):
                    result = original_score(module, batches, use_replay=use_replay)
                    rollout["rollout_log_probs"] = [value.detach().clone() for value in result]
                    score_hashes[str(step + 1)] = [tensor_record(value) for value in result]
                    return result

                def checked_loss(
                    parsed,
                    batch,
                    count,
                    logits,
                    *,
                    original_loss=original_loss,
                    original_log_probs=original_log_probs,
                    observed=observed,
                    **kwargs,
                ):
                    def checked_log_probs(*values, **options):
                        result = original_log_probs(*values, **options)
                        if not torch.is_grad_enabled() or not logits.requires_grad:
                            raise ValueError("Probe did not observe the actual grad-enabled policy forward")
                        for current, previous, mask in zip(
                            result["log_probs"], batch["log_probs"], batch["loss_masks"], strict=True
                        ):
                            delta = (current.detach().float() - previous.float()).abs()[mask.bool()]
                            observed[0] += delta.double().sum()
                            observed[1] += delta.numel()
                            observed[2] = torch.maximum(observed[2], delta.max().double())
                        return result

                    with mock.patch.object(losses, "get_log_probs_and_entropy", side_effect=checked_log_probs):
                        return original_loss(parsed, batch, count, logits, **kwargs)

                with (
                    mock.patch.object(worker, "_score", side_effect=scored),
                    mock.patch.object(actor.miles_loss, "loss_function", side_effect=checked_loss),
                    mock.patch.object(
                        actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())
                    ),
                ):
                    worker.train(step, None)
                dist.all_reduce(observed[:2])
                dist.all_reduce(observed[2:], op=dist.ReduceOp.MAX)
                check = {
                    "step": step + 1,
                    "active_tokens": int(observed[1]),
                    "mean_abs": float(observed[0] / observed[1]),
                    "max_abs": float(observed[2]),
                }
                score_checks.append(check)
                if (
                    not bool(torch.isfinite(observed).all())
                    or check["active_tokens"] != 52
                    or check["mean_abs"] > 1e-5
                    or check["max_abs"] > 1e-4
                ):
                    raise ValueError(f"No-grad scorer differs from actual grad-enabled policy loss: {check}")
                if step + 1 in (BOUNDARY, HORIZON):
                    snapshots[f"step{step + 1}"] = capture(worker, source)
                if phase == "split" and step + 1 == BOUNDARY:
                    if rank == 0:
                        source.save(step)
                    dist.barrier()
                    worker.save_model(step, force_sync=True)
                    worker.finalize_checkpoint(step)
                    snapshots["after_save2"] = capture(worker, source)
                    if verify_export:
                        exports["boundary"] = export_at_boundary()
                worker.clock.published()
        checkpoint_files = list((root / "split/core").rglob("*"))
        report = {
            "phase": phase,
            "rank": rank,
            "world": world,
            "pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - started,
            "snapshots": snapshots,
            "score_checks": score_checks,
            "score_hashes": score_hashes,
            "exports": exports,
            "save_timings": save_timings,
            "checkpoint_bytes": sum(p.stat().st_size for p in checkpoint_files if p.is_file()),
            "scheduler_horizon": HORIZON,
            "runtime_lock": runtime_lock(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        atomic_json(output / f"rank{rank}.json", report)
        if rank == 0:
            print(
                "DURABLE_CONTINUATION_PHASE_PASSED",
                json.dumps({k: report[k] for k in ("phase", "world", "elapsed_seconds", "checkpoint_bytes")}),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def audit(root, world):
    failures, comparisons = [], []
    for rank in range(world):
        phases = {
            name: json.loads((root / name / f"rank{rank}.json").read_text())
            for name in ("control", "split", "resumed")
        }
        if len({value["pid"] for value in phases.values()}) != 3:
            failures.append(f"rank{rank}: phases must run in fresh processes")
        if len({value["harness_sha256"] for value in phases.values()}) != 1:
            failures.append(f"rank{rank}: harness source changed between phases")
        if any(value["runtime_lock"] != phases["control"]["runtime_lock"] for value in phases.values()):
            failures.append(f"rank{rank}: runtime changed between phases")
        for name, value in phases.items():
            if value["world"] != world or value["rank"] != rank or value["scheduler_horizon"] != HORIZON:
                failures.append(f"rank{rank}/{name}: topology or horizon mismatch")
        first = phases["control"]["snapshots"]["step2"]["optimizer"]
        last = phases["control"]["snapshots"]["step4"]["optimizer"]
        for suffix in (".main", ".exp_avg", ".exp_avg_sq"):
            names = [name for name in first if name.endswith(suffix)]
            if not names or not any(first[name] != last.get(name) for name in names):
                failures.append(f"rank{rank}: no continuing {suffix} update signal")
        for phase, expected_steps in (("control", [1, 2, 3, 4]), ("split", [1, 2]), ("resumed", [3, 4])):
            checks = phases[phase]["score_checks"]
            if [item["step"] for item in checks] != expected_steps:
                failures.append(f"rank{rank}/{phase}: scorer checks do not cover every optimizer update")
            if any(
                item["active_tokens"] != 52 or not 0 <= item["mean_abs"] <= 1e-5 or not 0 <= item["max_abs"] <= 1e-4
                for item in checks
            ):
                failures.append(f"rank{rank}/{phase}: scorer and grad-forward probabilities differ")
        links = (
            ("control", "step2", "split", "step2"),
            ("split", "step2", "split", "after_save2"),
            ("split", "step2", "resumed", "restored2"),
            ("control", "step4", "resumed", "step4"),
        )
        for left, a, right, b in links:
            differences = compare_states(phases[left]["snapshots"][a], phases[right]["snapshots"][b])
            comparisons.append(
                {
                    "rank": rank,
                    "reference": f"{left}/{a}",
                    "actual": f"{right}/{b}",
                    "exact": not differences,
                    "differences": differences,
                }
            )
            failures.extend(f"rank{rank}/{left}/{a}->{right}/{b}: {item}" for item in differences)
    report = {
        "passed": not failures,
        "world": world,
        "comparisons": comparisons,
        "failures": failures,
        "limits": [
            "Exact local tensor bytes at fixed topology; no SGLang transport tested",
            "Synthetic responses and rewards; independent from the 100-update GSM8K run",
        ],
    }
    atomic_json(root / "audit.json", report)
    print(
        "DURABLE_CONTINUATION_AUDIT",
        json.dumps(
            {
                "passed": report["passed"],
                "world": world,
                "comparison_count": len(comparisons),
                "failure_count": len(failures),
                "first_failures": failures[:10],
            }
        ),
        flush=True,
    )
    if failures:
        raise ValueError("Durable continuation state mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--hf", type=Path)
    parser.add_argument("--phase", choices=("control", "split", "resumed"))
    parser.add_argument("--backend", choices=("torch", "flash_4"), default="torch")
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.hf)
    elif args.command == "run":
        if args.phase is None:
            parser.error("run requires --phase")
        run(args.root, args.phase, args.backend)
    else:
        audit(args.root, args.world)


if __name__ == "__main__":
    main()
