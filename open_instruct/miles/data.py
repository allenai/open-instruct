"""Preserve MILES sample semantics at the Core boundary."""

from typing import Any

import torch


def sample_batches(data: dict[str, Any], max_length: int) -> list[dict[str, Any]]:
    count = len(data["tokens"])
    if not count:
        raise ValueError("Cannot train an empty rollout batch")
    batches = []
    for index, tokens in enumerate(data["tokens"]):
        total = data["total_lengths"][index]
        response = data["response_lengths"][index]
        if tokens.ndim != 1 or len(tokens) != total or not 0 <= response < total <= max_length:
            raise ValueError("Invalid token/response lengths or Core context overflow")
        mask = data["loss_masks"][index]
        if mask.ndim != 1 or len(mask) != response or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("Response loss mask must contain one binary value per response token")
        batch = {key: [value[index]] for key, value in data.items() if isinstance(value, list) and len(value) == count}
        batch.update(tokens=tokens.unsqueeze(0), unconcat_tokens=[tokens], max_seq_lens=[total])
        if "dynamic_global_batch_size" in data:
            batch["dynamic_global_batch_size"] = data["dynamic_global_batch_size"]
        batches.append(batch)
    return batches


def policy_versions(batch: dict[str, Any]) -> list[int]:
    values = batch.get("weight_versions")
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(not isinstance(sample, (list, tuple)) or not sample for sample in values)
    ):
        raise ValueError("Every sample must carry its behavior policy version")
    versions = []
    for sample in values:
        for value in sample:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError("Invalid policy version")
            if isinstance(value, str) and (not value.isascii() or not value.isdigit()):
                raise ValueError("Invalid serialized policy version")
            if int(value) < 0:
                raise ValueError("Invalid negative policy version")
            versions.append(int(value))
    return versions


def router_routes(model, batch):
    routes = batch.get("rollout_routed_experts")
    if routes is None or len(routes) != 1:
        raise ValueError("Rollout router replay requires expert IDs for every sample")
    routes = routes[0]
    tokens = batch["tokens"].shape[1]
    if routes.ndim != 3 or routes.shape[0] != tokens - 1:
        raise ValueError("MILES replay must contain [tokens-1, layers, top_k] expert IDs")
    # MILES records routing for next-token prediction inputs. The last response
    # token has no scored successor; use a valid, deterministic assignment there.
    final = torch.arange(routes.shape[-1], device=routes.device).expand(1, routes.shape[1], -1)
    routes = torch.cat((routes, final), dim=0)
    mapping = {}
    for name, _ in model.named_modules():
        if name.endswith(".routed_experts_router"):
            parts = name.split(".")
            layer = int(parts[parts.index("blocks") + 1])
            mapping[name] = routes[:, layer].unsqueeze(0).long()
    return mapping
