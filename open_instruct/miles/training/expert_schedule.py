"""Replay-informed ordering before MILES' stride partition; dispatch routes never change.

Only complete optimizer blocks are permuted. Candidate bands are a heuristic;
acceptance always uses the Core packer, including world-wide pack equalization.
"""

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from open_instruct import logger_utils
from open_instruct.miles.training import expert_search, packing

logger = logger_utils.setup_logger(__name__)


@lru_cache(maxsize=8)
def model_layout(checkpoint, layer_stride=1):
    """Replay's layer axis includes dense blocks; score only routed blocks."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    if config.get("model_type") != "olmo3moe":
        raise ValueError("Expert scheduling currently requires an olmo3moe HF checkpoint")
    experts, layers, top_k = (config[key] for key in ("n_routed_experts", "num_hidden_layers", "num_experts_per_tok"))
    if any(type(n) is not int or n < 1 for n in (experts, layers, top_k, layer_stride)) or top_k > experts:
        raise ValueError("Invalid expert scheduling model dimensions")
    dense = config.get("dense_layers_indices") or []
    if any(type(i) is not int or not 0 <= i < layers for i in dense):
        raise ValueError("Invalid dense layer indices")
    selected = tuple(i for i in range(layers) if i not in dense)[::layer_stride]
    if not selected:
        raise ValueError("Expert scheduling requires routed layers")
    return experts, layers, top_k, selected


def destination_histogram(routes, length, *, num_experts, ep_degree, num_layers, top_k, layers):
    """One sample's [layer, destination rank] counts, including its synthetic tail.

    CPU rollout arrays stay on CPU. Trainer tensors are counted on their device;
    only the small histogram is copied to CPU, not the token-level routes.
    """
    if num_experts % ep_degree or not 1 <= top_k <= num_experts:
        raise ValueError("Invalid expert ownership or top-k")
    if routes is None:
        raise ValueError("Expert scheduling requires replay routes for every sample")
    expected = (length - 1, num_layers, top_k)
    local = num_experts // ep_degree
    tail = np.bincount(np.arange(top_k) // local, minlength=ep_degree)
    if isinstance(routes, torch.Tensor):
        if tuple(routes.shape) != expected or routes.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"Expected integer replay routes of shape {expected}")
        counts = []
        for layer in layers:
            ids = routes[:, layer].reshape(-1).long()
            if bool(((ids < 0) | (ids >= num_experts)).any()):
                raise ValueError("Replay expert ID outside model range")
            counts.append(torch.bincount(ids // local, minlength=ep_degree))
        return torch.stack(counts).cpu().numpy() + tail
    routes = np.asarray(routes)
    if routes.shape != expected or not np.issubdtype(routes.dtype, np.integer):
        raise ValueError(f"Expected integer replay routes of shape {expected}")
    counts = []
    for layer in layers:
        ids = routes[:, layer].reshape(-1)
        if np.any(ids < 0) or np.any(ids >= num_experts):
            raise ValueError("Replay expert ID outside model range")
        counts.append(np.bincount(ids.astype(np.int64) // local, minlength=ep_degree))
    return np.stack(counts) + tail


def schedule(order, lengths, *, world, max_tokens):
    """Exact [rank][pack][sample] membership from the trainer's unchanged packer."""
    if world < 1 or not lengths or len(lengths) % world or sorted(order) != list(range(len(lengths))):
        raise ValueError("Schedule requires a complete block permutation divisible by world")
    columns = [order[rank::world] for rank in range(world)]
    plans = [packing.plan([lengths[i] for i in column], max_tokens) for column in columns]
    count = max(map(len, plans))
    return [
        [[column[i] for i in pack] for pack in packing.equalize(plan, count)]
        for column, plan in zip(columns, plans, strict=True)
    ]


def dispatch_loads(order, lengths, histograms, *, world, ep_degree, max_tokens):
    """Counts indexed [EP group, pack, layer, destination rank]."""
    if ep_degree < 1 or world % ep_degree:
        raise ValueError("Expert parallel degree must divide world")
    membership = schedule(order, lengths, world=world, max_tokens=max_tokens)
    histograms = np.asarray(histograms)
    if histograms.ndim != 3 or histograms.shape[0] != len(lengths) or histograms.shape[2] != ep_degree:
        raise ValueError("Expected one [layer, EP slot] histogram per sample")
    local = np.stack([np.stack([histograms[pack].sum(axis=0) for pack in rank]) for rank in membership])
    return local.reshape(world // ep_degree, ep_degree, *local.shape[1:]).sum(axis=1)


def measurements(loads, attention=None):
    """JSON-safe dispatch skew and an absolute-work proxy, not predicted seconds."""
    peaks = loads.max(axis=-1)
    means = loads.mean(axis=-1)
    ratios = peaks.sum(axis=-1) / means.sum(axis=-1)
    result = {
        "packs_per_rank": int(loads.shape[1]),
        "dispatches": int(loads.shape[0] * loads.shape[1]),
        "skew_mean": float(ratios.mean()),
        "skew_max": float(ratios.max()),
        "max_slot_assignments": int(peaks.max()),
        "critical_work_proxy": int(peaks.max(axis=0).sum()),
    }

    if attention is not None:
        linear, quadratic = attention.max(axis=0).sum(axis=0)
        result.update(attention_token_work=int(linear), attention_pair_work=int(quadratic))
    return result


def measure(order, lengths, histograms, **kwargs):
    state = expert_search.Partition(order, lengths, np.asarray(histograms), **kwargs)
    return measurements(state.loads, state.attention)


def _candidate(lengths, histograms, *, world, ep_degree, max_tokens, sort_lengths):
    sequence = (
        sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)) if sort_lengths else list(range(len(lengths)))
    )
    rows = [sequence[i : i + world] for i in range(0, len(sequence), world)]
    starts = {pack[0] for pack in packing.plan([max(lengths[i] for i in row) for row in rows], max_tokens)}
    groups = world // ep_degree
    accumulated = np.zeros((groups, *histograms.shape[1:]), dtype=np.int64)
    tokens = [0] * world
    order = []
    for row_index, row in enumerate(rows):
        if row_index in starts:
            accumulated.fill(0)
        chosen = [[] for _ in range(groups)]
        # Integer concentration score: E * max - sum avoids floating point ties.
        for sample in sorted(
            row, key=lambda i: (-int((ep_degree * histograms[i].max(axis=1) - histograms[i].sum(axis=1)).sum()), i)
        ):
            group = min(
                (g for g in range(groups) if len(chosen[g]) < ep_degree),
                key=lambda g: (
                    int(((accumulated[g] + histograms[sample]).max(axis=1) - accumulated[g].max(axis=1)).sum()),
                    g,
                ),
            )
            chosen[group].append(sample)
            accumulated[group] += histograms[sample]
        placed = [None] * world
        for group, members in enumerate(chosen):
            ranks = list(range(group * ep_degree, (group + 1) * ep_degree))
            for sample in sorted(members, key=lambda i: (-lengths[i], i)):
                rank = min(ranks, key=lambda r: (tokens[r], r))
                ranks.remove(rank)
                tokens[rank] += lengths[sample]
                placed[rank] = sample
        order.extend(placed)
    return order


def plan_order(
    lengths,
    histograms,
    *,
    world,
    ep_degree,
    max_tokens,
    seed=0,
    max_proposals=1024,
    search_seconds=0.25,
    statistics=None,
):
    """Seeded bounded search with exact scoring and conservative identity fallback."""
    settings = dict(world=world, ep_degree=ep_degree, max_tokens=max_tokens)
    identity = list(range(len(lengths)))
    before = measure(identity, lengths, histograms, **settings)
    if not 1 < ep_degree < world:
        raise ValueError("Expert scheduling requires multiple EP groups and EP > 1")
    best, after = identity, before
    histograms = np.asarray(histograms, dtype=np.int64)
    for sort_lengths in (False, True):
        candidate = _candidate(lengths, histograms, **settings, sort_lengths=sort_lengths)
        score = measure(candidate, lengths, histograms, **settings)
        # Do not buy better normalized skew with more packs or greater peak work.
        guarded = (
            "packs_per_rank",
            "critical_work_proxy",
            "skew_mean",
            "skew_max",
            "max_slot_assignments",
            "attention_token_work",
            "attention_pair_work",
        )
        if all(score[key] <= after[key] for key in guarded) and any(score[key] < after[key] for key in guarded):
            best, after = candidate, score
    # Keep the already-qualified greedy result even if the bounded search cannot
    # improve it. Search starts from arrival, where useful pack structure exists.
    stats = statistics if statistics is not None else {}
    started = time.perf_counter()
    stats.update(seed=int(seed), guard_rejections=0)

    anchor = after.copy()
    normalizers = np.array([before[k] for k in ("critical_work_proxy", "attention_token_work", "attention_pair_work")])

    def ranking(score):
        work = np.array([score[k] for k in ("critical_work_proxy", "attention_token_work", "attention_pair_work")])
        return float((work / normalizers).sum()), score["critical_work_proxy"], score["skew_mean"]

    stats["guard_rejections_by_metric"] = dict.fromkeys(guarded, 0)

    def consider(candidate):
        nonlocal best, after
        score = measurements(candidate.loads, candidate.attention)
        failures = [key for key in guarded if score[key] > anchor[key]]
        if failures:
            stats["guard_rejections"] += 1
            for key in failures:
                stats["guard_rejections_by_metric"][key] += 1
            return
        if ranking(score) < ranking(after):
            best, after = candidate.order.tolist(), score

    state = expert_search.Partition(identity, lengths, histograms, **settings)
    expert_search.improve(
        state,
        seed=seed,
        max_proposals=max_proposals,
        deadline=started + search_seconds,
        consider=consider,
        statistics=stats,
    )
    stats.update(search_seconds=time.perf_counter() - started, order=best)
    return best, before, after


def reorder_samples(args, groups):
    """MILES' in-place producer callback, before reward normalization/DP partition.

    The outer lists are transport containers after this hook. Original prompt
    identity remains on each Sample.group_index, which pinned MILES uses to
    normalize rewards independently of adjacency. Never synthesize new IDs.
    """
    core = args.olmo_core
    if not core.expert_balanced_packing:
        return
    started = time.perf_counter()
    experts, layers, top_k, selected = model_layout(args.hf_checkpoint, core.expert_balance_layer_stride)
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    budget = core.packing_max_tokens or core.max_sequence_length
    samples = []
    for group in groups:
        if not isinstance(group, list) or not group or any(isinstance(s, list) for s in group):
            raise ValueError("Expert scheduling supports only single-turn prompt groups")
        if any(type(s.group_index) is not int or s.group_index < 0 or type(s.index) is not int for s in group):
            raise ValueError("Expert scheduling requires original group_index and sample index identities")
        if len({s.group_index for s in group}) != 1:
            raise ValueError("Input prompt group contains conflicting group identities")
        if any(s.rollout_id is not None for s in group):
            raise ValueError("Expert scheduling does not support compact or multi-turn rollouts")
        samples.extend(group)
    if len({s.index for s in samples}) != len(samples):
        raise ValueError("Expert scheduling requires unique sample indices")
    reordered = list(samples)
    records = []
    size = args.global_batch_size
    complete = len(samples) // size * size
    for start in range(0, complete, size):
        block = samples[start : start + size]
        lengths = [len(s.tokens) for s in block]
        histograms = [
            destination_histogram(
                s.rollout_routed_experts,
                n,
                num_experts=experts,
                ep_degree=core.expert_parallel_size,
                num_layers=layers,
                top_k=top_k,
                layers=selected,
            )
            for s, n in zip(block, lengths, strict=True)
        ]
        seed = int.from_bytes(
            hashlib.blake2b(json.dumps([s.index for s in block]).encode(), digest_size=8).digest(), "little"
        )
        statistics = {}
        order, before, after = plan_order(
            lengths,
            histograms,
            world=world,
            ep_degree=core.expert_parallel_size,
            max_tokens=budget,
            seed=seed,
            max_proposals=core.expert_balance_search_proposals,
            search_seconds=core.expert_balance_search_seconds / max(1, complete // size),
            statistics=statistics,
        )
        if sorted(order) != list(range(size)):
            raise ValueError("Expert schedule changed optimizer-step membership")
        reordered[start : start + size] = [block[i] for i in order]
        records.append(
            {
                "block": start // size,
                "before": before,
                "after": after,
                "reordered": order != list(range(size)),
                "search": statistics,
            }
        )
    # Only mutate after every block validates; preserve any tail that MILES trims.
    offset = 0
    for group in groups:
        count = len(group)
        group[:] = reordered[offset : offset + count]
        offset += count
    logger.info(
        "expert_schedule %s",
        json.dumps(
            {
                "event": "expert_schedule",
                "world": world,
                "ep_degree": core.expert_parallel_size,
                "layers": selected,
                "blocks": records,
                "untouched_tail_samples": len(samples) - complete,
                "planning_seconds": time.perf_counter() - started,
            },
            allow_nan=False,
            sort_keys=True,
        ),
    )


def local_loads(args, batches):
    """Actual trainer pack histograms; run under the actor's collective error guard."""
    core = args.olmo_core
    experts, layers, top_k, selected = model_layout(args.hf_checkpoint, core.expert_balance_layer_stride)
    packs = []
    for batch in batches:
        routes, lengths = batch.get("rollout_routed_experts"), batch["total_lengths"]
        if routes is None or len(routes) != len(lengths):
            raise ValueError("Replay routes missing from trainer pack")
        counts = [
            destination_histogram(
                r,
                n,
                num_experts=experts,
                ep_degree=core.expert_parallel_size,
                num_layers=layers,
                top_k=top_k,
                layers=selected,
            )
            for r, n in zip(routes, lengths, strict=True)
        ]
        packs.append(np.stack(counts).sum(axis=0))
    return torch.as_tensor(np.stack(packs), device=batches[0]["tokens"].device)


def realized_measurements(local, ep_degree):
    """Gather small histograms, summing source ranks within each actual EP group."""
    world = dist.get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    loads = torch.stack(gathered).reshape(world // ep_degree, ep_degree, *local.shape).sum(dim=1)
    return measurements(loads.cpu().numpy())
