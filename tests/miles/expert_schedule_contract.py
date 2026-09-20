"""Four-GPU EP2 qualification: producer permutation, scores, gradients and Adam state."""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
from unittest import mock

import ep_contract
import packing_contract
import torch
from miles.backends.training_utils import parallel
from miles.ray.rollout import rollout_data_conversion, train_data_conversion
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample
from torch import distributed as dist

from open_instruct.miles import actor, expert_schedule
from open_instruct.test_miles_expert_schedule import sample_groups


def rows_for(worker, rank, enabled):
    groups = sample_groups(8, Sample)
    for sample in sum(groups, []):
        sample.tokens = [(i + sample.index * 11) % 256 for i in range(17)]
        sample.response_length = 7
        sample.loss_mask = [1, 0, 1, 1, 1, 1, 1]
        sample.rollout_log_probs = [-1.0] * 7
        sample.rollout_routed_experts = sample.rollout_routed_experts[:1].repeat(16, axis=0)
    if enabled:
        expert_schedule.reorder_samples(worker.args, groups)
    flat, metadata = rollout_data_conversion.postprocess_rollout_data(worker.args, groups, {"dp_size": 4})
    converted = train_data_conversion.convert_samples_to_train_data(worker.args, flat, metadata, None, None)
    shard = train_data_conversion.split_train_data_by_dp_raw(worker.args, converted, dp_size=4)[rank]
    rows = train_data_conversion.process_rollout_data_shard(worker.args, shard)
    for key in ("tokens", "loss_masks", "rollout_log_probs", "rollout_routed_experts"):
        rows[key] = [torch.as_tensor(value, device="cuda") for value in rows[key]]
    return rows


def gathered_scores(worker, rows, steps, gloo):
    values = worker._score(worker.train_module, steps[0], use_replay=True)
    local = {i: v.cpu().tolist() for i, v in zip(rows["sample_indices"], values, strict=True)}
    gathered = [None] * 4
    dist.all_gather_object(gathered, local, group=gloo)
    return {key: torch.tensor(value) for shard in gathered for key, value in shard.items()}


def run(root, checkpointing):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 4
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
    report = {"passed": False, "world": world, "ep_degree": 2, "checkpointing": checkpointing, "arms": []}
    baseline = None
    try:
        for enabled in (False, True):
            worker, name = packing_contract.make_worker(
                root, world, True, False, checkpointing, expert_parallel_size=2, expert_balanced_packing=enabled
            )
            rows = rows_for(worker, rank, enabled)
            with mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=gloo):
                steps = worker._batch_steps(rows)
                scores = gathered_scores(worker, rows, steps, gloo)
                loads = expert_schedule.realized_measurements(expert_schedule.local_loads(worker.args, steps[0]), 2)
                gradients = []
                original_clip = worker.optimizer._clip_grad

                def capture_clip(gradients=gradients, worker=worker, original_clip=original_clip):
                    gradients.append(ep_contract.full_optimizer_state(worker, gradients=True))
                    return original_clip()

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
                assert any(float(v.abs().sum()) > 0 for v in gradients[0].values())
                errors = []
                if baseline is None:
                    baseline = (scores, gradients, state, loads)
                else:
                    assert scores.keys() == baseline[0].keys()
                    packing_contract.inspect_scores(
                        [baseline[0][i] for i in sorted(scores)], [scores[i] for i in sorted(scores)]
                    )
                    assert loads["skew_mean"] < baseline[3]["skew_mean"]
                    for expected, actual in zip([*baseline[1], baseline[2]], [*gradients, state], strict=True):
                        measured, error = ep_contract._state_errors(expected, actual)
                        assert not error, error
                        assert all(v["relative_l2_error"] < 0.05 for v in measured.values()), measured
                        errors.append(measured)
                contracts = [
                    json.loads(line)
                    for line in (root / name / f"training_contract_rank{rank}.jsonl").read_text().splitlines()
                ]
                replay = [r for r in contracts if r["event"] == "replay_routes"]
                assert replay and all(r["mismatches"] == 0 for r in replay)
                if enabled:
                    observed = [r for r in contracts if r["event"] == "expert_balance"]
                    assert len(observed) == 2 and all(r["skew_mean"] == loads["skew_mean"] for r in observed)
                report["arms"].append({"enabled": enabled, "loads": loads, "errors": errors})
            del worker, rows, steps, state, gradients
            gc.collect()
            torch.cuda.empty_cache()
        report["passed"] = True
    finally:
        (root / f"expert-schedule-ac{int(checkpointing)}-rank{rank}.json").write_text(
            json.dumps(report, allow_nan=False, indent=2) + "\n"
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    run(args.root, args.checkpointing)
