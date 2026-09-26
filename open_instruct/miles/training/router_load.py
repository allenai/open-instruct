"""Cheap update-level dispatch statistics from Core's forward-only router counters."""

import torch
from torch import distributed as dist


def snapshot(model):
    """Copy local pre-dispatch counts; never reset or mutate Core's accumulators.

    The actor resets these before training, after standalone/reference scoring.
    Core excludes backward recomputation through accumulate_metrics=False.
    """
    routers = [(name, module) for name, module in model.named_modules() if name.endswith("routed_experts_router")]
    if not routers:
        return None
    names, counts = [], []
    for name, router in routers:
        value = router.batch_size_per_expert.detach()
        if value.ndim != 1 or value.numel() != router.num_experts:
            raise ValueError(f"Invalid router dispatch counter: {name}")
        names.append(name)
        counts.append(value.to(dtype=torch.int64))
    return names, torch.stack(counts)


def _statistics(counts):
    """Last axis is experts; retain layer/replica axes until after normalization."""
    values = counts.double()
    mean = values.mean(dim=-1)
    denominator = mean.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "max_expert_load": counts.amax(dim=-1),
        "dead_experts": (counts == 0).sum(dim=-1),
        "max_mean_load_ratio": values.amax(dim=-1) / denominator,
        "load_cv": values.std(dim=-1, correction=0) / denominator,
    }


def measurements(names, rank_counts, ep_degree):
    """Summarize [source rank, layer, expert] counts over one optimizer update.

    Contiguous EP ranks handle different source tokens; sum them into each replica
    before computing replica imbalance. Sum replicas for the global histogram.
    A global dead expert means an unused (layer, expert) pair, not an expert ID
    pooled across different layers. All-zero layers have ratio/CV zero.
    """
    if rank_counts.ndim != 3 or rank_counts.shape[1] != len(names) or not rank_counts.shape[2]:
        raise ValueError("Expected [rank, layer, expert] counts matching the router names")
    world, layers, experts = rank_counts.shape
    if not layers or not world or ep_degree < 1 or world % ep_degree or bool((rank_counts < 0).any()):
        raise ValueError("Invalid router counts or expert parallel degree")
    replica_counts = rank_counts.reshape(world // ep_degree, ep_degree, layers, experts).sum(dim=1)
    global_counts = replica_counts.sum(dim=0)
    global_stats, replica_stats = _statistics(global_counts), _statistics(replica_counts)
    per_layer = {
        name: {
            "assignments": int(global_counts[i].sum()),
            "experts": experts,
            **{key: values[i].item() for key, values in global_stats.items()},
        }
        for i, name in enumerate(names)
    }
    summary = {
        "moe/max_expert_load": int(global_stats["max_expert_load"].max()),
        "moe/dead_experts": int(global_stats["dead_experts"].sum()),
        "moe/dead_experts_max_per_layer": int(global_stats["dead_experts"].max()),
        "moe/load_cv_mean": float(global_stats["load_cv"].mean()),
        "moe/load_cv_max": float(global_stats["load_cv"].max()),
        "moe/max_mean_load_ratio": float(global_stats["max_mean_load_ratio"].max()),
        "moe/replica_load_cv_max": float(replica_stats["load_cv"].max()),
        "moe/replica_max_mean_load_ratio": float(replica_stats["max_mean_load_ratio"].max()),
        "moe/replica_dead_experts_max_per_layer": int(replica_stats["dead_experts"].max()),
    }
    return {"summary": summary, "layers": per_layer, "replicas": world // ep_degree}


def collect(local, ep_degree):
    """Gather only the small layer/expert histograms, never token-level routes."""
    if local is None:  # Dense model: no metrics and no added collective.
        return None
    names, counts = local
    if dist.is_initialized():
        gathered = [torch.empty_like(counts) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, counts)
    else:
        gathered = [counts]
    return measurements(names, torch.stack(gathered).cpu(), ep_degree)
