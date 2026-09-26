"""Four CPU ranks, two EP groups: executed packs agree with rollout predictions."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from open_instruct.miles.training import contract, data, expert_schedule, packing
from open_instruct.test_miles_expert_schedule import histograms, hook_args, sample_groups


def exercise(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        args = hook_args(Path(output) / str(rank))
        args.save = output
        groups = sample_groups()
        original = sum(groups, [])
        expert_schedule.reorder_samples(args, groups)
        flat = sum(groups, [])
        order = [s.index for s in flat]
        orders = [None] * 4
        dist.all_gather_object(orders, order)
        assert all(other == order for other in orders)
        local = flat[rank::4]
        raw = dict(
            tokens=[torch.tensor(s.tokens) for s in local],
            total_lengths=[len(s.tokens) for s in local],
            response_lengths=[s.response_length for s in local],
            loss_masks=[torch.tensor(s.loss_mask) for s in local],
            rollout_routed_experts=[torch.from_numpy(s.rollout_routed_experts) for s in local],
        )
        # Execute the same rank-local plan and world-wide equalization as _batch_steps.
        plan = packing.plan(raw["total_lengths"], 12)
        count = torch.tensor(len(plan))
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        batches = [packing.combine(data.sample_batches(raw, 12), ids) for ids in packing.equalize(plan, int(count))]
        measured = expert_schedule.realized_measurements(expert_schedule.local_loads(args, batches), 2)
        expected = expert_schedule.measure(order, [6] * 16, histograms(original), world=4, ep_degree=2, max_tokens=12)
        assert measured == {k: expected[k] for k in measured}
        json.loads(contract.record(args, {"event": "expert_balance", **measured}))
        norm = contract.step_normalization(batches, 16)
        assert norm.samples == 16 and norm.model_tokens == 96 and norm.active_tokens == 32
        # Simulate changed physical pack boundaries as well, including unequal local counts.
        lengths = [2, 2, 9, 9] * 4
        membership = expert_schedule.schedule(list(range(16)), lengths, world=4, max_tokens=12)
        counts = np.stack([np.array([[2 * n, 0]]) for n in lengths])
        local = torch.tensor(np.stack([counts[pack].sum(axis=0) for pack in membership[rank]]))
        measured = expert_schedule.realized_measurements(local, 2)
        expected = expert_schedule.measure(list(range(16)), lengths, counts, world=4, ep_degree=2, max_tokens=12)
        assert measured == {k: expected[k] for k in measured}
    finally:
        dist.destroy_process_group()


def test_four_rank_expert_schedule(tmp_path):
    for rank in range(4):
        (tmp_path / str(rank)).mkdir()
    mp.spawn(exercise, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path / f"training_contract_rank{rank}.jsonl").exists() for rank in range(4))
