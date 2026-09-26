"""Incremental search must agree with the independent full-packing scorer."""

import json

import numpy as np
import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.training import expert_schedule, expert_search


def test_incremental_swaps_match_full_rescore_across_pack_changes():
    rng = np.random.default_rng(19)
    modes = set()
    for world, ep_degree in ((4, 2), (8, 2), (8, 4)):
        lengths = rng.integers(1, 13, world * 4).tolist()
        histograms = np.stack([rng.multinomial(n * 2, np.ones(ep_degree) / ep_degree, size=3) for n in lengths])
        settings = dict(world=world, ep_degree=ep_degree, max_tokens=16)
        state = expert_search.Partition(list(range(len(lengths))), lengths, histograms, **settings)
        for _ in range(100):
            old_order, old_loads = state.order.copy(), state.loads.copy()
            a, b = rng.choice(len(lengths), 2, replace=False)
            candidate, mode = state.swap(int(a), int(b))
            modes.add(mode)
            np.testing.assert_array_equal(state.order, old_order)
            np.testing.assert_array_equal(state.loads, old_loads)
            expected = expert_schedule.dispatch_loads(candidate.order.tolist(), lengths, histograms, **settings)
            np.testing.assert_array_equal(candidate.loads, expected)
            membership = expert_schedule.schedule(candidate.order.tolist(), lengths, world=world, max_tokens=16)
            attention = np.array(
                [
                    [[sum(lengths[i] for i in pack), sum(lengths[i] ** 2 for i in pack)] for pack in rank]
                    for rank in membership
                ]
            )
            np.testing.assert_array_equal(candidate.attention, attention)
            assert (
                expert_search.cost(candidate.loads, 1)[0]
                == expert_schedule.measurements(expected)["critical_work_proxy"]
            )
            # Mix accepted/rejected proposals to catch stale cache/aliasing bugs.
            if rng.random() < 0.5:
                state = candidate
    assert modes == {"fixed", "columns", "global"}


def test_near_equal_lengths_do_not_imply_fixed_membership():
    lengths = [56, 56, 45, 44, 44, 44, 1, 1]
    counts = np.array([[[n, n]] for n in lengths])
    state = expert_search.Partition(list(range(8)), lengths, counts, world=4, ep_degree=2, max_tokens=100)
    # Across columns, 45 -> 44 can merge two packs while 44 -> 45 separates another.
    found = False
    for a in range(8):
        for b in range(a + 1, 8):
            if abs(lengths[a] - lengths[b]) == 1:
                candidate, mode = state.swap(a, b)
                if mode != "fixed":
                    found = True
                    np.testing.assert_array_equal(
                        candidate.loads,
                        expert_schedule.dispatch_loads(
                            candidate.order.tolist(), lengths, counts, world=4, ep_degree=2, max_tokens=100
                        ),
                    )
    assert found


def test_search_finds_improvement_missed_by_greedy():
    # Heterogeneous loads and unequal lengths make greedy band estimates inaccurate.
    rng = np.random.default_rng(7)
    lengths = rng.integers(2, 13, 32).tolist()
    counts = np.stack([rng.multinomial(n * 4, [0.3, 0.7], size=3) for n in lengths])
    settings = dict(world=8, ep_degree=2, max_tokens=24)
    greedy, before, old = expert_schedule.plan_order(lengths, counts, **settings, max_proposals=0)
    stats = {}
    order, _, after = expert_schedule.plan_order(
        lengths, counts, **settings, seed=17, max_proposals=1024, search_seconds=100, statistics=stats
    )
    assert after["critical_work_proxy"] < old["critical_work_proxy"]
    assert order != greedy
    assert all(after[k] <= old[k] for k in before)
    repeat, _, _ = expert_schedule.plan_order(
        lengths, counts, **settings, seed=17, max_proposals=stats["iterations"], search_seconds=100
    )
    assert order == repeat
    json.dumps(stats, allow_nan=False)


def test_expired_search_returns_greedy_and_records_stop():
    lengths = [6] * 16
    counts = np.array([[[11, 1] if i % 4 < 2 else [1, 11]] for i in range(16)])
    settings = dict(world=4, ep_degree=2, max_tokens=12)
    expected = expert_schedule.plan_order(lengths, counts, **settings, max_proposals=0)
    stats = {}
    actual = expert_schedule.plan_order(lengths, counts, **settings, search_seconds=0, statistics=stats)
    assert actual == expected
    assert stats["stop"] == "deadline" and stats["proposals"] == 0


@pytest.mark.parametrize(
    "name,value",
    [
        ("expert_balance_search_seconds", -1),
        ("expert_balance_search_seconds", float("nan")),
        ("expert_balance_search_proposals", -1),
        ("expert_balance_search_proposals", 1.5),
    ],
)
def test_search_limits_are_validated(name, value):
    with pytest.raises(ValueError):
        CoreConfig(**{name: value}).validate()


def test_smoothing_orders_an_equal_exact_work_plateau():
    # Neither the max nor either attention proxy changes; a near-bottleneck does.
    counts = np.array([[[10, 2]], [[0, 6]], [[5, 0]], [[0, 1]]])
    state = expert_search.Partition(list(range(4)), [4] * 4, counts, world=4, ep_degree=2, max_tokens=4)
    candidate, _ = state.swap(1, 3)
    normalizers = state.work()
    old, new = expert_search.objective(state, normalizers), expert_search.objective(candidate, normalizers)
    assert new[0] == old[0]
    assert new[1] < old[1]


def test_attention_uses_sum_of_document_squares_not_square_of_pack_length():
    state = expert_search.Partition(
        list(range(8)), [2, 3, 4, 5, 6, 7, 8, 9], np.ones((8, 1, 2), dtype=int), world=4, ep_degree=2, max_tokens=20
    )
    np.testing.assert_array_equal(state.attention[:, 0], [[8, 40], [10, 58], [12, 80], [14, 106]])
    score = expert_schedule.measurements(state.loads, state.attention)
    assert score["attention_token_work"] == 14 and score["attention_pair_work"] == 106
