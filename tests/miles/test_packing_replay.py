"""MILES packing preserves the Core adapter's metadata and replay tails."""

import numpy as np
import pytest
import torch
from expert_schedule_helpers import histograms, sample_groups
from miles.backends.core_utils import packing
from torch import nn

from open_instruct.miles.training import data


def fixture():
    lengths = [5, 8, 6, 7]
    return {
        "tokens": [torch.arange(n) + 20 * i for i, n in enumerate(lengths)],
        "total_lengths": lengths,
        "response_lengths": [2] * 4,
        "loss_masks": [torch.tensor([1, 0])] * 4,
        "weight_versions": [[str(i)] for i in range(4)],
        "rewards": [float(i) for i in range(4)],
        "rollout_routed_experts": [torch.full((n - 1, 1, 2), 2, dtype=torch.int32) for n in lengths],
    }


def test_packed_metadata_and_replay_have_one_tail_per_sample():
    raw = fixture()
    samples = data.sample_batches(raw, 16)
    batch = packing.combine(samples, [0, 1])
    assert torch.equal(batch["tokens"], torch.cat(raw["tokens"][:2])[None])
    assert batch["doc_lens"].tolist() == [[5, 8]] and batch["max_doc_lens"] == [8]
    assert batch["weight_versions"] == [["0"], ["1"]] and batch["rewards"] == [0.0, 1.0]
    assert batch["total_lengths"] == [5, 8] and batch["max_seq_lens"] is None
    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": nn.Module()})
    model.blocks["0"].routed_experts_router = nn.Identity()
    routes = data.router_routes(model, batch)["blocks.0.routed_experts_router"]
    assert routes.shape == (1, 13, 2)
    assert routes[0, 4].tolist() == routes[0, 12].tolist() == [0, 1]
    assert routes[0, 5].tolist() == [2, 2]
    raw["rollout_routed_experts"][1] = torch.zeros(8, 1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="per sample"):
        data.router_routes(model, packing.combine(data.sample_batches(raw, 16), [0, 1]))


def test_histograms_match_trainer_replay_including_each_tail():
    samples = sum(sample_groups(), [])[:3]
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleDict({"1": torch.nn.Module()})
    model.blocks["1"].routed_experts_router = torch.nn.Identity()
    batch = dict(
        tokens=torch.zeros(1, 18),
        total_lengths=[6] * 3,
        rollout_routed_experts=[torch.from_numpy(s.rollout_routed_experts) for s in samples],
    )
    replay = data.router_routes(model, batch)["blocks.1.routed_experts_router"]
    expected = torch.bincount(replay.flatten() // 2, minlength=2).numpy()
    np.testing.assert_array_equal(np.stack(histograms(samples)).sum(axis=0)[0], expected)
    for sample in samples:
        cpu = histograms([sample])[0]
        sample.rollout_routed_experts = torch.from_numpy(sample.rollout_routed_experts)
        np.testing.assert_array_equal(histograms([sample])[0], cpu)
