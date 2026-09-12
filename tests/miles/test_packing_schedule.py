"""Rank-local pack counts align without crossing optimizer steps or adding tokens."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from open_instruct.miles import actor
from open_instruct.miles.config import CoreConfig


def exercise(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    actor.distributed_utils.init_gloo_group()
    try:
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = SimpleNamespace(
            global_batch_size=8, olmo_core=CoreConfig(sequence_packing=True, max_sequence_length=16)
        )
        lengths = [2, 2, 2, 2, 9, 9, 2, 2] if rank == 0 else [9, 9, 2, 2, 2, 2, 2, 2]
        rows = {
            "tokens": [torch.full((n,), i) for i, n in enumerate(lengths)],
            "total_lengths": lengths,
            "response_lengths": [1] * 8,
            "loss_masks": [torch.ones(1)] * 8,
            "weight_versions": [[str(i // 4)] for i in range(8)],
        }
        steps = worker._batch_steps(rows)
        assert len(steps) == 2
        for index, batches in enumerate(steps):
            assert len(batches) == 2
            assert sum(len(b["total_lengths"]) for b in batches) == 4
            assert [int(t[0]) for b in batches for t in b["unconcat_tokens"]] == list(range(index * 4, index * 4 + 4))
            assert all(v == [str(index)] for b in batches for v in b["weight_versions"])
            assert sum(b["tokens"].numel() for b in batches) == sum(lengths[index * 4 : index * 4 + 4])
        # A rank-local malformed sample must fail on both ranks before any forward.
        if rank == 1:
            rows["total_lengths"][0] += 1
        try:
            worker._batch_steps(rows)
        except RuntimeError as error:
            assert "rank 1" in str(error)
        else:
            raise AssertionError("bad input did not fail collectively")
        Path(output, str(rank)).write_text("passed")
    finally:
        dist.destroy_process_group()


def test_distributed_packing_schedule(tmp_path):
    mp.spawn(exercise, args=(str(tmp_path / "group"), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path / str(rank)).read_text() == "passed" for rank in [0, 1])
