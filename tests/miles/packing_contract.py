"""Fixed-input KDA/attention/latent-MoE packing qualification on one or two GPUs."""

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ep_contract
import torch
from miles.backends.training_utils import parallel
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist

from open_instruct.miles import actor, data, models, scheduler
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.state import PolicyClock


def bootstrap(root):
    ep_contract.bootstrap(root)
    path = root / "hf/config.json"
    hf = json.loads(path.read_text())
    hf["use_rope"] = True
    path.write_text(json.dumps(hf, indent=2) + "\n")


def rollout(rank, world):
    lengths = [17, 87, 19, 13, 23, 89, 29, 11]
    rows = {
        key: []
        for key in (
            "tokens",
            "total_lengths",
            "response_lengths",
            "loss_masks",
            "rewards",
            "weight_versions",
            "rollout_routed_experts",
        )
    }
    for i in range(rank, len(lengths), world):
        length = lengths[i]
        response = min(7 + i, length - 2)
        tokens = (torch.arange(length, device="cuda") + 11 * i) % 256
        routes = torch.stack([tokens[:-1] % 4, (tokens[:-1] + 1) % 4], dim=-1)
        routes = routes[:, None, :].expand(-1, 2, -1).contiguous()
        mask = torch.ones(response, device="cuda")
        mask[1] = 0
        for key, value in {
            "tokens": tokens,
            "total_lengths": length,
            "response_lengths": response,
            "loss_masks": mask,
            "rewards": float((i // world) % 2),
            "weight_versions": ["0"],
            "rollout_routed_experts": routes,
        }.items():
            rows[key].append(value)
    return rows


def make_worker(root, world, packed, auxiliary, checkpointing):
    name = f"ep{world}-pack{int(packed)}-aux{int(auxiliary)}-ac{int(checkpointing)}"
    run = RunConfig(
        CoreConfig(
            expert_parallel_size=world,
            attention_backend="flash_4",
            max_sequence_length=128,
            sequence_packing=packed,
            activation_checkpointing=checkpointing,
            diagnostic_interval=1,
            replay_diagnostics=True,
            scoring_check_interval=50,
            router_aux_loss_weight=0.01 if auxiliary else 0.0,
            router_z_loss_weight=1e-5 if auxiliary else 0.0,
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            global_batch_size=8,
            rollout_batch_size=2,
            n_samples_per_prompt=4,
            num_rollout=2,
            actor_num_gpus_per_node=world,
            debug_train_only=True,
            save=str(root / name),
            rollout_global_dataset=True,
            prompt_data=str(root / "prompts.jsonl"),
            lr=1e-4,
            clip_grad=1e9,
            calculate_per_token_loss=True,
            use_rollout_routing_replay=True,
            use_miles_router=True,
            use_tis=True,
            tis_clip=1.1,
            tis_clip_low=0.9,
            seed=173,
        ),
    )
    sys.argv = ["packing-contract", *run.arguments()]
    args = arguments.parse_args()
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = args
    worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
    worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
    worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
    worker.clock = PolicyClock()
    worker.clock.published()
    worker.ref_module = None
    worker._heartbeat = SimpleNamespace(bump=lambda: None)
    return worker, name


def inspect_scores(reference, actual):
    delta = torch.cat([(a - b).abs().float() for a, b in zip(reference, actual, strict=True)])
    report = {"mean_abs": float(delta.mean()), "max_abs": float(delta.max()), "tokens": delta.numel()}
    assert report["mean_abs"] < 0.005 and report["max_abs"] < 0.05, report
    return report


def run(root, checkpointing):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
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
    report, baseline = {"passed": False, "world": world, "checkpointing": checkpointing, "arms": []}, None
    try:
        for packed, auxiliary in [(False, False), (True, False), (True, True)]:
            worker, name = make_worker(root, world, packed, auxiliary, checkpointing)
            rows = rollout(rank, world)
            with mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo):
                steps = worker._batch_steps(rows)
                assert len(steps) == 1
                if packed:
                    assert len(steps[0]) < len(rows["tokens"])
                reference = worker._score(worker.train_module, data.sample_batches(rows, 128), use_replay=True)
                current = worker._score(worker.train_module, steps[0], use_replay=True)
                agreement = inspect_scores(reference, current)
                # Vary the preceding document while holding shape/routes fixed.
                # Other documents must remain independent of its attention/KDA state.
                if packed:
                    changed = {**rows, "tokens": [v.clone() for v in rows["tokens"]]}
                    changed["tokens"][0] = (changed["tokens"][0] + 37) % 256
                    mutated = worker._score(worker.train_module, worker._batch_steps(changed)[0], use_replay=True)
                    isolation = inspect_scores(current[1:], mutated[1:])
                else:
                    isolation = None
                gradients = []
                original_clip = worker.optimizer._clip_grad

                def capture_clip(gradients=gradients, worker=worker, original_clip=original_clip):
                    gradients.append(ep_contract.full_optimizer_state(worker, gradients=True))
                    return original_clip()

                times = []
                with mock.patch.object(worker.optimizer, "_clip_grad", side_effect=capture_clip):
                    for update in range(2):
                        rows["weight_versions"] = [[str(update)] for _ in rows["tokens"]]
                        rows["rollout_log_probs"] = worker._score(worker.train_module, steps[0], use_replay=True)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        with mock.patch.object(
                            actor.miles_data, "get_rollout_data", return_value=(rows, contextlib.nullcontext())
                        ):
                            worker.train(update, None)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - started)
                        worker.clock.published()
                assert any(float(v.abs().sum()) > 0 for v in gradients[0].values()), "policy fixture has no gradient"
                state = ep_contract.full_optimizer_state(worker)
                assert worker.clock.completed_steps == 2
                errors = None
                if not packed:
                    baseline = (gradients, state)
                elif not auxiliary:
                    errors = []
                    for expected, actual in zip([*baseline[0], baseline[1]], [*gradients, state], strict=True):
                        measured, error = ep_contract._state_errors(expected, actual)
                        assert not error, error
                        assert all(v["relative_l2_error"] < 0.05 for v in measured.values()), measured
                        errors.append(measured)
                assert all(torch.isfinite(v).all() for v in state.values())
                arm = dict(
                    name=name,
                    packed=packed,
                    auxiliary=auxiliary,
                    packs=len(steps[0]),
                    samples=len(rows["tokens"]),
                    scores=agreement,
                    isolation=isolation,
                    step_seconds=times,
                    errors=errors,
                )
                report["arms"].append(arm)
                path = Path(worker.args.save) / f"training_contract_rank{rank}.jsonl"
                contracts = [json.loads(line) for line in path.read_text().splitlines()]
                replay = [r for r in contracts if r["event"] == "replay_routes"]
                assert replay and all(r["mismatches"] == 0 for r in replay)
                training = [r for r in replay if r["phase"] == "training"]
                assert training
                if checkpointing:
                    assert all(c["entered"] >= 2 for r in training for c in r["layers"].values())
                events = [r for r in contracts if r["event"] == "optimizer"]
                assert [r["scoring_pass"] for r in events] == ["checked", "skipped"]
                print(json.dumps(arm), flush=True)
            del worker, state, gradients, steps, rows
            gc.collect()
            torch.cuda.empty_cache()
        report["passed"] = True
    finally:
        (root / f"packing-ep{world}-ac{int(checkpointing)}-rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "run"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpointing", action="store_true")
    options = parser.parse_args()
    if options.command == "bootstrap":
        bootstrap(options.root)
    else:
        run(options.root, options.checkpointing)
