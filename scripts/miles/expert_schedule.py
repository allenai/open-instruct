"""Historical offline scheduling prototype; production uses miles.expert_schedule.

This helper retains the earlier within-replica skew scorer and greedy experiments.
It does not model the global slowest replica or attention cost and must not be used
as a production throughput predictor. The qualified planner lives in
open_instruct.miles.expert_schedule and includes those additional work proxies.
"""

import numpy as np

from open_instruct import logger_utils
from open_instruct.miles import packing

logger = logger_utils.setup_logger(__name__)


def local_expert_count(num_experts: int, ep_degree: int) -> int:
    """Experts owned by one expert-parallel slot; Core assigns contiguous blocks."""
    if num_experts < 1 or ep_degree < 1 or num_experts % ep_degree:
        raise ValueError(f"{num_experts} experts cannot be divided across {ep_degree} expert-parallel slots")
    return num_experts // ep_degree


def destination_histogram(routes, *, num_experts: int, ep_degree: int, layer_stride: int = 1) -> np.ndarray:
    """Dispatch counts per [layer, expert-parallel slot] for one sample's replayed routes.

    `routes` is the sample's [tokens - 1, layers, top_k] expert ids. The trainer appends one
    synthetic tail row per document (`data.router_routes`), so it is counted here too; leaving it
    out makes the prediction disagree with the realized dispatch by one row per sample.
    """
    array = np.asarray(routes)
    if array.ndim != 3:
        raise ValueError(f"Replay routes must be [tokens - 1, layers, top_k]; found shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Replay routes must be integer expert ids; found dtype {array.dtype}")
    _, layers, top_k = array.shape
    tail = np.broadcast_to(np.arange(top_k, dtype=array.dtype), (1, layers, top_k))
    array = np.concatenate([array, tail], axis=0)
    if array.min(initial=0) < 0 or array.max(initial=0) >= num_experts:
        raise ValueError("Replay routes contain expert ids outside the model's expert range")
    per_slot = array // local_expert_count(num_experts, ep_degree)
    selected = range(0, layers, max(1, layer_stride))
    return np.stack([np.bincount(per_slot[:, layer].reshape(-1), minlength=ep_degree) for layer in selected])


def _pack_membership(lengths: list[int], max_tokens: int, count: int) -> list[list[int]]:
    """Exactly what the trainer will pack for one rank, including the equalizing splits."""
    return packing.equalize(packing.plan(lengths, max_tokens), count)


def dispatch_loads(order, lengths, histograms, *, world: int, ep_degree: int, max_tokens: int) -> list[np.ndarray]:
    """Per-dispatch [layer, slot] loads that this order actually produces.

    One dispatch is one pack index inside one expert-parallel group. Groups are contiguous blocks
    of `ep_degree` ranks, and the all-to-all sums every source rank in the group, so a sample's
    contribution depends on its group and pack index, not on which rank inside the group holds it.
    """
    columns = [[order[position] for position in range(rank, len(order), world)] for rank in range(world)]
    plans = [packing.plan([lengths[index] for index in column], max_tokens) for column in columns]
    count = max(len(plan) for plan in plans)
    membership = [
        [
            [column[index] for index in pack]
            for pack in _pack_membership([lengths[i] for i in column], max_tokens, count)
        ]
        for column, plan in zip(columns, plans, strict=True)
    ]
    loads = []
    for group_start in range(0, world, ep_degree):
        ranks = range(group_start, min(group_start + ep_degree, world))
        for pack_index in range(count):
            total = np.zeros_like(histograms[0])
            for rank in ranks:
                for sample in membership[rank][pack_index]:
                    total = total + histograms[sample]
            loads.append(total)
    return loads


def skew(loads: list[np.ndarray]) -> dict:
    """Straggler cost of a set of dispatches: slowest slot over the mean, summed across layers."""
    values = []
    for load in loads:
        per_layer_max = float(load.max(axis=1).sum())
        per_layer_mean = float(load.mean(axis=1).sum())
        values.append(per_layer_max / per_layer_mean if per_layer_mean > 0 else 1.0)
    if not values:
        return dict(dispatches=0, skew_mean=1.0, skew_max=1.0)
    return dict(dispatches=len(values), skew_mean=float(np.mean(values)), skew_max=float(np.max(values)))


def measure(order, lengths, histograms, *, world: int, ep_degree: int, max_tokens: int) -> dict:
    """JSON-safe skew of the dispatches an order produces, scored through the real packer."""
    return skew(dispatch_loads(order, lengths, histograms, world=world, ep_degree=ep_degree, max_tokens=max_tokens))


def plan_order(
    lengths,
    histograms,
    *,
    world: int,
    ep_degree: int,
    max_tokens: int,
    row_order: str = "arrival",
    compare: bool = True,
) -> list[int]:
    """Permute one optimizer step so its dispatches are evenly loaded.

    Position `j * world + r` becomes rank `r`'s `j`-th sample, matching MILES' stride partition.
    Within a row, samples go to the expert-parallel group whose slowest slot grows least, most
    concentrated samples first. Deterministic: integer arithmetic, explicit tie-breaks, no
    randomness.

    With `compare` (the default) the candidate is scored against the arrival order and only
    returned if it is measurably better, so planning can never make a dispatch worse.

    `row_order` selects which samples share a row. "arrival" keeps the producer's order, which
    preserves the natural mixing of document types across a dispatch. "length" groups similar
    lengths so each rank's packer advances in step, at the cost of that mixing: on heterogeneous
    data it measurably raises dispatch skew, so it is not the default.
    """
    total = len(lengths)
    if total != len(histograms):
        raise ValueError("Every sample needs one replay histogram")
    if world < 1 or total % world:
        raise ValueError(f"A block of {total} samples cannot be split across {world} ranks")
    if ep_degree < 1 or world % ep_degree:
        raise ValueError(f"World size {world} is not divisible by expert-parallel degree {ep_degree}")
    if row_order == "length":
        sequence = sorted(range(total), key=lambda index: (-lengths[index], index))
    elif row_order == "arrival":
        sequence = list(range(total))
    else:
        raise ValueError(f"Unknown row order {row_order!r}; use 'arrival' or 'length'")
    rows = [sequence[start : start + world] for start in range(0, total, world)]
    groups = world // ep_degree
    order = [None] * total
    accumulated = [np.zeros_like(histograms[0]) for _ in range(groups)]
    tokens = [0] * world
    # Balance is per dispatch, so the accumulator resets at every band: rows that the packer will
    # place at the same pack index. Accumulating across the whole block instead balances group
    # totals and can leave individual dispatches worse than the order it started from.
    proxies = [max(lengths[index] for index in row) for row in rows]
    starts = {pack[0] for pack in packing.plan(proxies, max_tokens)}
    for row_index, row in enumerate(rows):
        if row_index in starts:
            accumulated = [np.zeros_like(histograms[0]) for _ in range(groups)]
        concentration = sorted(
            row,
            key=lambda index: (-float((histograms[index].max(axis=1) - histograms[index].mean(axis=1)).sum()), index),
        )
        capacity = [ep_degree] * groups
        chosen = {group: [] for group in range(groups)}
        for sample in concentration:
            group = min(
                (g for g in range(groups) if capacity[g]),
                key=lambda g: (_growth(accumulated[g], histograms[sample]), g),
            )
            capacity[group] -= 1
            chosen[group].append(sample)
            accumulated[group] = accumulated[group] + histograms[sample]
        for group, members in chosen.items():
            ranks = list(range(group * ep_degree, (group + 1) * ep_degree))
            for sample in sorted(members, key=lambda index: (-lengths[index], index)):
                rank = min(ranks, key=lambda r: (tokens[r], r))
                ranks.remove(rank)
                tokens[rank] += lengths[sample]
                order[row_index * world + rank] = sample
    if any(position is None for position in order):
        raise ValueError("Expert-aware ordering failed to place every sample")
    if not compare:
        return order
    # Moving a sample to another rank also shifts that rank's pack boundaries, so a candidate can
    # pair worse-matched samples in a dispatch than the order it started from. The scorer is
    # exact, so keep the arrival order unless the candidate is measurably better.
    scored = dict(world=world, ep_degree=ep_degree, max_tokens=max_tokens)
    arrival = list(range(total))
    if (
        measure(order, lengths, histograms, **scored)["skew_mean"]
        < measure(arrival, lengths, histograms, **scored)["skew_mean"]
    ):
        return order
    return arrival


def _growth(accumulated: np.ndarray, histogram: np.ndarray) -> float:
    """How much this sample would raise the slowest slot of a group, summed across layers."""
    return float((accumulated + histogram).max(axis=1).sum() - accumulated.max(axis=1).sum())
