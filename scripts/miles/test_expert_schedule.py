"""Expert-aware ordering: permutation safety, exact scoring, and measured balance gains."""

import numpy as np
import pytest
import torch
from scripts.miles import expert_schedule

from open_instruct.miles.training import data, packing

EXPERTS = 64
TOP_K = 4
LAYERS = 3


def routes(expert_ids, tokens):
    """Replay routes that send every token of a sample to a fixed set of experts."""
    row = np.array(expert_ids, dtype=np.int32).reshape(1, 1, TOP_K)
    return np.broadcast_to(row, (tokens - 1, LAYERS, TOP_K)).copy()


def block(world=4, rows=3, hot=None, clustered=False):
    """One optimizer step: half the samples route low, half route high.

    `clustered` puts all the low-routing samples in the first half of each row, which is
    adversarial for the arrival order: the stride then hands one expert-parallel group only
    low-routing samples and the other only high-routing ones.
    """
    lengths, histograms = [], []
    for index in range(world * rows):
        low = (index % world) < world // 2 if clustered else index % 2 == 0
        experts = [0, 1, 2, 3] if low else [EXPERTS - 4, EXPERTS - 3, EXPERTS - 2, EXPERTS - 1]
        lengths.append(100 + (index % 5))
        histograms.append(
            expert_schedule.destination_histogram(
                routes(hot if hot else experts, lengths[-1]), num_experts=EXPERTS, ep_degree=4
            )
        )
    return lengths, histograms


def test_histogram_counts_the_trainer_tail_row():
    tokens = 7
    histogram = expert_schedule.destination_histogram(routes([0, 1, 2, 3], tokens), num_experts=EXPERTS, ep_degree=2)
    assert histogram.shape == (LAYERS, 2)
    # Every token routes to slot 0, and the appended tail row spreads over the first top_k experts.
    assert histogram[0].tolist() == [tokens * TOP_K, 0]


def test_histogram_matches_router_routes_dispatch():
    """The predicted histogram must equal what data.router_routes actually replays."""
    tokens = 9
    sample = routes([0, 1, EXPERTS - 2, EXPERTS - 1], tokens)
    predicted = expert_schedule.destination_histogram(sample, num_experts=EXPERTS, ep_degree=2)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleDict({str(layer): torch.nn.Module() for layer in range(LAYERS)})
    for layer in range(LAYERS):
        model.blocks[str(layer)].routed_experts_router = torch.nn.Identity()
    batch = {
        "tokens": torch.zeros(1, tokens, dtype=torch.long),
        "total_lengths": [tokens],
        "rollout_routed_experts": [torch.tensor(sample)],
    }
    replayed = data.router_routes(model, batch)
    local = expert_schedule.local_expert_count(EXPERTS, 2)
    for layer, ids in enumerate(replayed[name] for name in sorted(replayed)):
        counts = np.bincount((ids.reshape(-1).numpy() // local), minlength=2)
        assert counts.tolist() == predicted[layer].tolist()


def test_histogram_rejects_malformed_routes():
    with pytest.raises(ValueError):
        expert_schedule.destination_histogram(np.zeros((4, 2), dtype=np.int32), num_experts=EXPERTS, ep_degree=2)
    with pytest.raises(ValueError):
        expert_schedule.destination_histogram(
            routes([0, 1, 2, 3], 5).astype(np.float32), num_experts=EXPERTS, ep_degree=2
        )
    with pytest.raises(ValueError):
        expert_schedule.destination_histogram(routes([0, 1, 2, 3], 5), num_experts=3, ep_degree=2)


def test_order_is_a_deterministic_permutation_with_equal_rank_counts():
    lengths, histograms = block(world=4, rows=3)
    order = expert_schedule.plan_order(lengths, histograms, world=4, ep_degree=2, max_tokens=512)
    assert sorted(order) == list(range(len(lengths)))
    assert order == expert_schedule.plan_order(lengths, histograms, world=4, ep_degree=2, max_tokens=512)
    for rank in range(4):
        assert len(order[rank::4]) == 3


def test_order_respects_the_token_budget_on_every_rank():
    lengths, histograms = block(world=4, rows=3)
    order = expert_schedule.plan_order(lengths, histograms, world=4, ep_degree=2, max_tokens=256)
    for rank in range(4):
        column = [lengths[index] for index in order[rank::4]]
        for pack in packing.plan(column, 256):
            assert sum(column[index] for index in pack) <= 256


def test_planning_improves_measured_dispatch_balance():
    """The group-assignment lever works when the arrival order leaves headroom."""
    lengths, histograms = block(world=4, rows=4, clustered=True)
    identity = list(range(len(lengths)))
    order = expert_schedule.plan_order(lengths, histograms, world=4, ep_degree=2, max_tokens=256)
    scored = dict(world=4, ep_degree=2, max_tokens=256)
    before = expert_schedule.measure(identity, lengths, histograms, **scored)
    after = expert_schedule.measure(order, lengths, histograms, **scored)
    assert after["skew_mean"] < before["skew_mean"]
    assert after["dispatches"] == before["dispatches"]


def test_scoring_uses_real_pack_membership_not_assumed_bands():
    """Equal pack counts can still hide different membership; the scorer must pack for real."""
    lengths = [56, 45, 44, 56, 44, 44]
    histograms = [
        expert_schedule.destination_histogram(routes([0, 1, 2, 3], length), num_experts=EXPERTS, ep_degree=2)
        for length in lengths
    ]
    order = list(range(len(lengths)))
    columns = [[order[position] for position in range(rank, len(order), 2)] for rank in range(2)]
    plans = [packing.plan([lengths[i] for i in column], 100) for column in columns]
    assert len(plans[0]) == len(plans[1])
    assert [len(pack) for pack in plans[0]] != [len(pack) for pack in plans[1]]
    loads = expert_schedule.dispatch_loads(order, lengths, histograms, world=2, ep_degree=2, max_tokens=100)
    # One dispatch per pack index, and the totals must cover every sample's tokens exactly once.
    assert len(loads) == len(plans[0])
    assert sum(float(load.sum()) for load in loads) == sum(float(h.sum()) for h in histograms)


def test_planning_leaves_an_already_balanced_order_alone():
    """Interleaved arrivals are balanced already; planning must not make them worse."""
    lengths, histograms = block(world=4, rows=4)
    scored = dict(world=4, ep_degree=2, max_tokens=256)
    before = expert_schedule.measure(list(range(len(lengths))), lengths, histograms, **scored)
    after = expert_schedule.measure(
        expert_schedule.plan_order(lengths, histograms, **scored), lengths, histograms, **scored
    )
    assert after["skew_mean"] <= before["skew_mean"] + 1e-9


def test_single_expert_group_keeps_band_composition_only():
    lengths, histograms = block(world=2, rows=3)
    order = expert_schedule.plan_order(lengths, histograms, world=2, ep_degree=2, max_tokens=256)
    assert sorted(order) == list(range(len(lengths)))


def test_order_rejects_blocks_that_do_not_divide():
    lengths, histograms = block(world=4, rows=2)
    with pytest.raises(ValueError):
        expert_schedule.plan_order(lengths[:-1], histograms[:-1], world=4, ep_degree=2, max_tokens=256)
    with pytest.raises(ValueError):
        expert_schedule.plan_order(lengths, histograms, world=4, ep_degree=3, max_tokens=256)
