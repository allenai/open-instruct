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
    lengths = batch.get("total_lengths", [batch["tokens"].numel()])
    if routes is None or len(routes) != len(lengths):
        raise ValueError("Rollout router replay requires expert IDs for every sample")
    padded = []
    for sample, length in zip(routes, lengths, strict=True):
        if not isinstance(sample, torch.Tensor) or sample.dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Replay expert IDs must be integer tensors")
        if sample.ndim != 3 or sample.shape[0] != length - 1:
            raise ValueError("MILES replay must contain [tokens-1, layers, top_k] expert IDs per sample")
        # Every document has its own unscored final token, including interior
        # documents in a pack. Never use the following sample's first assignment.
        final = torch.arange(sample.shape[-1], device=sample.device).expand(1, sample.shape[1], -1)
        padded.append(torch.cat((sample, final), dim=0))
    routes = torch.cat(padded, dim=0)
    if routes.shape[0] != batch["tokens"].numel():
        raise ValueError("Replay routes do not cover packed tokens")
    mapping = {}
    for name, _ in model.named_modules():
        if name.endswith(".routed_experts_router"):
            parts = name.split(".")
            layer = int(parts[parts.index("blocks") + 1])
            if layer >= routes.shape[1]:
                raise ValueError("Replay layer axis does not cover every routed block")
            mapping[name] = routes[:, layer].unsqueeze(0).long()
    if not mapping:
        raise ValueError("Router replay requires routed MoE blocks")
    return mapping


def score_agreement(rollout: dict[str, Any]) -> torch.Tensor:
    """Return absolute-difference sum and active-token count for rank reduction."""
    scores = rollout["log_probs"]
    behavior = rollout.get("rollout_log_probs")
    masks = rollout["loss_masks"]
    if behavior is None or not scores or len(scores) != len(behavior) or len(scores) != len(masks):
        raise ValueError("Behavior and training log probabilities must cover the same samples")
    result = torch.zeros(2, dtype=torch.float64, device=scores[0].device)
    for current, previous, mask in zip(scores, behavior, masks, strict=True):
        previous = torch.as_tensor(previous, device=current.device, dtype=torch.float32)
        if current.ndim != 1 or current.shape != previous.shape or current.shape != mask.shape:
            raise ValueError("Behavior/training score and response-mask shapes differ")
        active = mask.bool()
        current, previous = current[active].float(), previous[active]
        if not bool(torch.isfinite(current).all() & torch.isfinite(previous).all()):
            raise ValueError("Non-finite active-token log probability")
        result[0] += (current - previous).abs().double().sum()
        result[1] += active.sum()
    return result


def validate_score_agreement(stats: torch.Tensor, limit: float | None) -> float:
    """Validate the globally reduced active-token mean before any optimizer step."""
    difference = float(stats[0] / stats[1].clamp_min(1))
    if limit is not None and difference > limit:
        raise ValueError(f"Train/rollout logprob difference {difference:.6f} exceeds {limit:.6f}")
    return difference
