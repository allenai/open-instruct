"""Document-aware router auxiliaries without changing packed model execution.

The default delegates to Core unchanged. Nondefault objectives use the same
selected count source and differentiable scores, with explicit global denominators.
Metadata travels in the auxiliary tuple so backward recomputation cannot reuse a
subsequent microbatch's document statistics.
"""

import contextlib
import contextvars
import dataclasses
import types

import torch


@dataclasses.dataclass(frozen=True)
class Batch:
    lengths: tuple[int, ...]
    response_denominator: float


_CURRENT = contextvars.ContextVar("miles_router_objective_batch", default=None)


@contextlib.contextmanager
def batch_context(batch, model=None):
    lengths = tuple(int(n) for n in batch.get("total_lengths", [batch["tokens"].numel()]))
    if not lengths or min(lengths) <= 0 or sum(lengths) != batch["tokens"].numel():
        raise ValueError("Router documents must partition the forwarded model tokens")
    denominator = float(batch.get("aux_loss_response_div_factor", len(lengths)))
    if denominator <= 0:
        raise ValueError("Router response denominator must be positive")
    metadata = Batch(lengths, denominator)
    token = _CURRENT.set(metadata)
    # Non-reentrant activation checkpointing may recompute on an autograd worker
    # thread whose ContextVars are empty. Core keeps this scope open through the
    # entire microbatch backward, so router-local metadata has the same lifetime
    # as native replay state and is visible to those threads. Microbatches execute
    # serially; restore even on failure and retain immutable metadata in each tuple.
    routers = getattr(model, "_miles_objective_routers", ())
    previous = [getattr(router, "_miles_batch", None) for router in routers]
    for router in routers:
        router._miles_batch = metadata
    try:
        yield
    finally:
        for router, value in zip(routers, previous, strict=True):
            router._miles_batch = value
        _CURRENT.reset(token)


def document_losses(
    scores,
    logits,
    indices,
    lengths,
    *,
    top_k,
    token_denominator,
    response_denominator,
    grouping="sequence",
    reduction="token",
    z_reduction="token",
):
    """Return coefficient-free losses; counts are nondifferentiable expert IDs.

    Denominators are global totals divided by the gradient-averaging world size.
    Response weighting includes an inner model-token mean for each document.
    """
    if scores.ndim != 3 or scores.shape[0] != 1 or logits.shape != scores.shape:
        raise ValueError("Document router objectives require one unpadded packed instance")
    if sum(lengths) != scores.shape[1] or min(lengths) <= 0:
        raise ValueError("Invalid router document lengths")
    if indices.shape != (*scores.shape[:2], top_k):
        raise ValueError("Router assignments do not cover the model tokens")
    if token_denominator is None or token_denominator <= 0 or response_denominator <= 0:
        raise ValueError("Router objectives require explicit positive update denominators")
    if (
        grouping not in ("pack", "sequence")
        or reduction not in ("token", "response")
        or z_reduction not in ("token", "response")
    ):
        raise ValueError("Unknown document router objective")
    experts = scores.shape[-1]
    probabilities = scores[0]
    with torch.no_grad():
        assignments = indices[0].detach()
        counts = [
            torch.bincount(ids.reshape(-1), minlength=experts).to(scores.dtype) for ids in assignments.split(lengths)
        ]
        pack_fraction = torch.stack(counts).sum(0) / (top_k * sum(lengths))
    lb_terms, z_terms = [], []
    for length, p, z, count in zip(
        lengths, probabilities.split(lengths), logits[0].split(lengths), counts, strict=True
    ):
        fraction = count / (top_k * length) if grouping == "sequence" else pack_fraction
        weight = length / token_denominator if reduction == "token" else 1 / response_denominator
        lb_terms.append(experts * (fraction * p.mean(0)).sum() * weight)
        if z_reduction == "response":
            z_terms.append(z.logsumexp(-1).square().mean() / response_denominator)
    z_loss = logits.logsumexp(-1).square().sum() / token_denominator if z_reduction == "token" else sum(z_terms)
    return sum(lb_terms), z_loss


def _forward(self, *args, **kwargs):
    result = self._miles_native_router_forward(*args, **kwargs)
    weights, indices, counts, info = result
    if info is None or not self.training or not torch.is_grad_enabled():
        return result
    batch = getattr(self, "_miles_batch", None) or _CURRENT.get()
    if batch is None:
        raise RuntimeError("Document router forward requires a batch context covering backward")
    return weights, indices, counts, (*info, indices, batch)


def _compute(
    self,
    scores,
    logits,
    counts,
    batched_counts,
    loss_div_factor,
    indices=None,
    batch=None,
    pending_global_counts=None,
    *,
    accumulate_metrics=True,
    reduced_global_counts=None,
):
    if not self.training or not torch.is_grad_enabled():
        return None
    if indices is None or batch is None or pending_global_counts is not None or reduced_global_counts is not None:
        raise RuntimeError("Unsupported document router auxiliary call")
    if str(self.gating_function) == "sigmoid":
        scores = scores / scores.sum(dim=-1, keepdim=True)
    if getattr(self, "lb_loss_count_source", "dispatch") == "current":
        # Recompute from this forward's scores, not mutable replay state. This
        # changes only balancing statistics; actual dispatch and metrics stay intact.
        with torch.no_grad():
            indices = self.get_top_k(scores.detach())[1]
    grouping, reduction, z_reduction = self._miles_objective
    lb, z = document_losses(
        scores,
        logits,
        indices,
        batch.lengths,
        top_k=self.top_k,
        token_denominator=loss_div_factor,
        response_denominator=batch.response_denominator,
        grouping=grouping,
        reduction=reduction,
        z_reduction=z_reduction,
    )
    auxiliary = None
    if self.lb_loss_weight is not None:
        if accumulate_metrics:
            self.load_balancing_loss += lb.detach()
        auxiliary = self.lb_loss_weight * lb
    if self.z_loss_weight is not None:
        if accumulate_metrics:
            self.z_loss += z.detach()
        weighted = self.z_loss_weight * z
        auxiliary = weighted if auxiliary is None else auxiliary + weighted
    if accumulate_metrics:
        self.batch_size_per_expert += counts
        if self.bias_gamma is not None:
            self.score_bias_batch_size_per_expert += counts
    return auxiliary


def install(model, options):
    """Select per-router auxiliary reduction; preserve Core's default bit-for-bit."""
    objective = (options.router_aux_loss_grouping, options.router_aux_loss_reduction, options.router_z_loss_reduction)
    native_objective = objective == ("pack", "token", "token")
    if native_objective and options.router_aux_count_source == "dispatch":
        return
    routers = [(name, module) for name, module in model.named_modules() if name.endswith("routed_experts_router")]
    if not routers:
        raise ValueError("Router objective controls require a Core MoE model")
    if native_objective:
        return
    model._miles_objective_routers = tuple(router for _, router in routers)
    for name, router in routers:
        if getattr(router, "global_load_balancing", False) or getattr(router, "orth_loss_weight", None) is not None:
            raise ValueError(f"Document router objective does not support global balancing or orthogonal loss: {name}")
        if router.tp_mesh is not None or router.cp_mesh is not None:
            raise ValueError("Document router objectives currently require trainer TP=CP=1")
        if hasattr(router, "_miles_objective"):
            raise ValueError("Document router objective was already installed")
        router._miles_objective = objective
        router._miles_native_router_forward = router.forward
        router.forward = types.MethodType(_forward, router)
        router.compute_aux_loss = types.MethodType(_compute, router)
