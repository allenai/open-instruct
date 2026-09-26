"""Native fixed-input qualification of document router objectives and repacking."""

import argparse
import contextlib
import gc
import json
import os
import time
from pathlib import Path
from unittest import mock

import ep_contract
import packing_contract
import torch
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist

from open_instruct.miles.training import actor


def run(root, checkpointing, count_source="dispatch"):
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
    report = dict(passed=False, world=world, checkpointing=checkpointing, count_source=count_source, arms=[])
    try:
        for reduction in ("token", "response"):
            reference = None
            for packed in (False, True):
                arm_root = root / f"{count_source}-{reduction}-{int(packed)}-ac{int(checkpointing)}"
                arm_root.mkdir(exist_ok=True)
                for name in ("hf", "prompts.jsonl"):
                    target = arm_root / name
                    if not target.exists():
                        target.symlink_to(root / name)
                worker, name = packing_contract.make_worker(
                    arm_root,
                    world,
                    packed,
                    True,
                    checkpointing,
                    router_aux_loss_grouping="sequence",
                    router_aux_count_source=count_source,
                    router_aux_loss_reduction=reduction,
                )
                gradients = []
                original_clip = worker.optimizer._clip_grad

                def capture_clip(worker=worker, gradients=gradients, original_clip=original_clip):
                    gradients.append(ep_contract.full_optimizer_state(worker, gradients=True))
                    return original_clip()

                rows = packing_contract.rollout(rank, world)
                started = time.perf_counter()
                with mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo):
                    steps = worker._batch_steps(rows)
                    with mock.patch.object(worker.optimizer, "_clip_grad", side_effect=capture_clip):
                        for update in range(2):
                            rows["weight_versions"] = [[str(update)] for _ in rows["tokens"]]
                            rows["rollout_log_probs"] = worker._score(worker.train_module, steps[0], use_replay=True)
                            with mock.patch.object(
                                actor.miles_data, "get_rollout_data", return_value=(rows, contextlib.nullcontext())
                            ):
                                worker.train(update, None)
                            worker.clock.published()
                    state = ep_contract.full_optimizer_state(worker)
                assert all(torch.isfinite(v).all() for v in state.values())
                assert any(float(v.abs().sum()) > 0 for v in gradients[0].values())
                actual = [*gradients, state]
                errors = []
                if reference is None:
                    reference = actual
                else:
                    for expected, measured in zip(reference, actual, strict=True):
                        metrics, error = ep_contract._state_errors(expected, measured)
                        assert not error, error
                        assert all(v["relative_l2_error"] < 0.05 for v in metrics.values()), metrics
                        errors.append(metrics)
                contracts = [
                    json.loads(line)
                    for line in (Path(worker.args.save) / f"training_contract_rank{rank}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                replay = [r for r in contracts if r["event"] == "replay_routes"]
                assert replay and all(r["mismatches"] == 0 for r in replay)
                if checkpointing:
                    assert all(
                        c["entered"] >= 2 for r in replay if r["phase"] == "training" for c in r["layers"].values()
                    )
                events = [r for r in contracts if r["event"] == "optimizer"]
                assert [r["scoring_pass"] for r in events] == ["checked", "skipped"]
                report["arms"].append(
                    dict(reduction=reduction, packed=packed, errors=errors, seconds=time.perf_counter() - started)
                )
                print(json.dumps(report["arms"][-1]), flush=True)
                del worker, state, gradients, rows, steps, actual
                gc.collect()
                torch.cuda.empty_cache()
        report["passed"] = True
    finally:
        (root / f"router-{count_source}-ep{world}-ac{int(checkpointing)}-rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "run"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--count-source", choices=["dispatch", "current"], default="dispatch")
    args = parser.parse_args()
    if args.command == "bootstrap":
        packing_contract.bootstrap(args.root)
    else:
        run(args.root, args.checkpointing, args.count_source)
