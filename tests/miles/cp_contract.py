"""Compare Core EP2/CP1 and EP2/CP2 policy gradients and Adam moments.

Uses the existing hybrid fixture and MILES' actual GRPO loss and CP helpers.
Uneven document lengths, masked response tokens and replay cross shard boundaries.
The fixture includes Small-style Scalable-Softmax and nonunit per-head Q/K gains.
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ep_contract
import torch
from miles.backends.core_utils import actor, context_parallel, models, scheduler
from miles.backends.core_utils.publication.state import PolicyClock
from miles.backends.training_utils import parallel
from miles.utils import arguments
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


def run(root, cp, *, checkpointing, token_average):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    try:
        group, trivial = GroupInfo(rank, world, dist.group.WORLD, gloo), GroupInfo(0, 1, None)
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
        config = RunConfig(
            CoreConfig(
                expert_parallel_size=world,
                attention_backend="flash_4",
                max_sequence_length=128,
                activation_checkpointing=checkpointing,
                sequence_packing=True,
                packing_max_tokens=128,
                diagnostic_interval=1,
                scoring_check_interval=1,
                replay_diagnostics=True,
                router_aux_loss_weight=0.0,
                router_z_loss_weight=0.0,
            ),
            dict(
                hf_checkpoint=str(root / "hf"),
                global_batch_size=4,
                rollout_batch_size=1,
                n_samples_per_prompt=4,
                num_rollout=1,
                actor_num_gpus_per_node=world,
                context_parallel_size=cp,
                debug_train_only=True,
                save=str(root / f"cp{cp}-ac{int(checkpointing)}-token{int(token_average)}"),
                rollout_global_dataset=True,
                prompt_data=str(root / "prompts.jsonl"),
                lr=1e-4,
                clip_grad=1e9,
                calculate_per_token_loss=token_average,
                use_rollout_routing_replay=True,
                use_miles_router=True,
            ),
        )
        sys.argv = ["cp-contract", *config.arguments()]
        args = arguments.parse_args()
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = args
        with mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo):
            worker.train_module, worker.hf_config, worker.model_config = models.build_train_module(args)
            worker.train_parallel_config = context_parallel.configure(worker.train_module)
        worker.model, worker.optimizer = worker.train_module.model, worker.train_module.optim
        worker.lr_scheduler = scheduler.CoreLRScheduler(args, worker.optimizer)
        worker._observed_response_tokens = 0
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        worker._heartbeat = SimpleNamespace(bump=lambda: None)
        state = parallel.get_parallel_state()
        rollout = ep_contract.rollout_for(state.effective_dp.rank, state.effective_dp.size, "policy")
        # Uneven lengths exercise padding and the logit/label
        # shift at CP boundaries. Both layouts use identical recorded routes.
        all_routes = torch.load(root / "routes.pt", weights_only=True)
        indices = list(range(state.effective_dp.rank, 4, state.effective_dp.size))
        rollout["rollout_routed_experts"] = [all_routes[i].cuda() for i in indices]
        packs = worker._batch_steps(rollout)
        rollout["rollout_log_probs"] = worker._score(
            worker.train_module, [batch for step in packs for batch in step], use_replay=True
        )
        with (
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo),
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
        ):
            worker.train(0, None)
        result = ep_contract.full_optimizer_state(worker)
        if rank == 0:
            torch.save(result, root / f"cp{cp}-ac{int(checkpointing)}-token{int(token_average)}.pt")
    finally:
        dist.destroy_process_group()


def compare(root):
    report = {"passed": True, "relative_l2_tolerance": 0.05, "comparisons": []}
    for token_average in (False, True):
        baseline = torch.load(root / f"cp1-ac0-token{int(token_average)}.pt", weights_only=True)
        signals, error = ep_contract._moment_signals(baseline)
        if error or any(values["l2"] == 0 for values in signals.values()):
            raise AssertionError(f"Disconnected baseline gradients: {signals}, {error}")
        for cp, checkpointing in ((1, True), (2, False), (2, True)):
            actual = torch.load(
                root / f"cp{cp}-ac{int(checkpointing)}-token{int(token_average)}.pt", weights_only=True
            )
            errors, error = ep_contract._state_errors(baseline, actual)
            passed = error is None and all(v["relative_l2_error"] < 0.05 for v in errors.values())
            report["passed"] &= passed
            report["comparisons"].append(
                dict(
                    cp=cp,
                    checkpointing=checkpointing,
                    token_average=token_average,
                    passed=passed,
                    errors=errors,
                    error=error,
                )
            )
    (root / "cp-contract.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    if not report["passed"]:
        raise AssertionError("CP gradients/Adam states differ from CP1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--token-average", action="store_true")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.root)
    else:
        run(args.root, args.cp, checkpointing=args.checkpointing, token_average=args.token_average)


if __name__ == "__main__":
    main()
