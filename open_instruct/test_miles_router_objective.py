"""Independent references for document-aware auxiliary normalization and integration."""

import contextvars
from types import SimpleNamespace

import pytest
import torch
from olmo_core import config as core_config
from olmo_core.nn.moe.v2 import router as core_router
from torch.utils import checkpoint

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.training import contract, router_load, router_objective


def reference(logits, indices, lengths, token_denominator, response_denominator, grouping, reduction, z_reduction):
    probabilities = logits.softmax(-1)[0]
    experts, topk = logits.shape[-1], indices.shape[-1]
    result = logits.sum() * 0
    z_loss = logits.sum() * 0
    offset = 0
    for length in lengths:
        selected = indices[0, offset : offset + length] if grouping == "sequence" else indices[0]
        count = torch.tensor([(selected == expert).sum().item() for expert in range(experts)], dtype=logits.dtype)
        fraction = count / selected.numel()
        denominator = token_denominator if reduction == "token" else response_denominator * length
        zden = token_denominator if z_reduction == "token" else response_denominator * length
        for position in range(offset, offset + length):
            result = result + experts * (fraction * probabilities[position]).sum() / denominator
            z_loss = z_loss + logits[0, position].logsumexp(-1).square() / zden
        offset += length
    assert topk > 0
    return result, z_loss


@pytest.mark.parametrize("grouping", ["pack", "sequence"])
@pytest.mark.parametrize("reduction", ["token", "response"])
@pytest.mark.parametrize("z_reduction", ["token", "response"])
def test_loss_and_gradient_independent_reference(grouping, reduction, z_reduction):
    logits = torch.randn(
        1, 11, 4, generator=torch.Generator().manual_seed(39), dtype=torch.float64, requires_grad=True
    )
    indices = torch.tensor([[[0, 1]] * 2 + [[1, 3]] * 6 + [[0, 2]] * 3])
    kwargs = dict(
        token_denominator=27.0,
        response_denominator=7.0,
        grouping=grouping,
        reduction=reduction,
        z_reduction=z_reduction,
    )
    actual = router_objective.document_losses(logits.softmax(-1), logits, indices, (2, 6, 3), top_k=2, **kwargs)
    expected = reference(logits, indices, (2, 6, 3), **kwargs)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(a, logits, retain_graph=True)[0], torch.autograd.grad(e, logits, retain_graph=True)[0]
        )


@pytest.mark.parametrize("reduction", ["token", "response"])
def test_document_objective_invariant_to_repacking(reduction):
    logits = torch.randn(1, 11, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64, requires_grad=True)
    indices = logits.detach().topk(2, dim=-1).indices
    kwargs = dict(top_k=2, token_denominator=11.0, response_denominator=3.0, grouping="sequence", reduction=reduction)
    together = router_objective.document_losses(logits.softmax(-1), logits, indices, (2, 6, 3), **kwargs)
    pieces = [
        router_objective.document_losses(z.softmax(-1), z, ids, (n,), **kwargs)
        for z, ids, n in zip(logits.split((2, 6, 3), dim=1), indices.split((2, 6, 3), dim=1), (2, 6, 3), strict=True)
    ]
    separated = tuple(sum(p[k] for p in pieces) for k in range(2))
    for a, b in zip(together, separated, strict=True):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(
            torch.autograd.grad(a, logits, retain_graph=True)[0], torch.autograd.grad(b, logits, retain_graph=True)[0]
        )


def test_canary_direction_and_unchanged_z():
    logits = torch.tensor([[[2.0, 0.0]] * 2 + [[0.0, 2.0]] * 6], dtype=torch.float64, requires_grad=True)
    indices = logits.detach().argmax(-1, keepdim=True)
    results = []
    for grouping, reduction in [("pack", "token"), ("sequence", "token"), ("sequence", "response")]:
        lb, z = router_objective.document_losses(
            logits.softmax(-1),
            logits,
            indices,
            (2, 6),
            top_k=1,
            token_denominator=8,
            response_denominator=2,
            grouping=grouping,
            reduction=reduction,
        )
        results.append((0.01 * lb.item(), torch.autograd.grad(0.01 * lb, logits, retain_graph=True)[0], z))
    assert results[0][0] == pytest.approx(0.011903985389889411)
    assert results[1][0] == results[2][0] == pytest.approx(0.017615941559557646)
    assert results[0][1][0, 0, 0] < 0 < results[1][1][0, 0, 0]
    assert not torch.equal(results[1][1], results[2][1])
    assert torch.equal(results[0][2], results[1][2]) and torch.equal(results[0][2], results[2][2])


def test_rank_average_global_denominators():
    logits = torch.randn(1, 8, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    indices = logits.argmax(-1, keepdim=True)
    for reduction in ["token", "response"]:
        kwargs = dict(top_k=1, grouping="sequence", reduction=reduction)
        full = router_objective.document_losses(
            logits.softmax(-1), logits, indices, (2, 6), token_denominator=8, response_denominator=2, **kwargs
        )
        ranks = [
            router_objective.document_losses(
                z.softmax(-1), z, ids, (n,), token_denominator=4, response_denominator=1, **kwargs
            )
            for z, ids, n in zip(logits.split((2, 6), dim=1), indices.split((2, 6), dim=1), (2, 6), strict=True)
        ]
        for k in range(2):
            torch.testing.assert_close(sum(x[k] for x in ranks) / 2, full[k])


class Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.tp_mesh = self.cp_mesh = self.orth_loss_weight = None
        self.global_load_balancing = False
        self.bias_gamma = None
        self.batch_size_per_expert = torch.zeros(2)
        self.gating_function = "softmax"
        self.top_k = 1
        self.lb_loss_weight, self.z_loss_weight = 0.01, 1e-5
        self.load_balancing_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)

    def forward(self, x):
        logits = x @ self.weight
        scores = logits.softmax(-1)
        # Deliberately disagree with current top-k to verify replay is respected.
        ids = 1 - scores.argmax(-1, keepdim=True)
        counts = torch.bincount(ids.reshape(-1), minlength=2)
        return scores.gather(-1, ids), ids, counts, (scores, logits, counts, counts[None], 8.0)

    def compute_aux_loss(self, *args, **kwargs):
        return torch.tensor(42.0)


def test_adapter_preserves_forward_and_replayed_counts():
    model = torch.nn.Module()
    model.routed_experts_router = Router()
    router = model.routed_experts_router
    x = torch.tensor([[[2.0, 0.0]] * 2 + [[0.0, 2.0]] * 6])
    expected = router(x)
    router_objective.install(
        model, CoreConfig(router_aux_loss_grouping="sequence", router_aux_loss_reduction="response")
    )
    with pytest.raises(RuntimeError, match="batch context"):
        router(x)
    batch = dict(tokens=torch.zeros(1, 8), total_lengths=[2, 6], aux_loss_response_div_factor=2)
    with router_objective.batch_context(batch):
        actual = router(x)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        loss = router.compute_aux_loss(*actual[3])
        ref_lb, ref_z = reference(actual[3][1], actual[1], (2, 6), 8.0, 2.0, "sequence", "response", "token")
        torch.testing.assert_close(loss, 0.01 * ref_lb + 1e-5 * ref_z)
        loss.backward()
        assert router.weight.grad.abs().sum() > 0
    assert router_objective._CURRENT.get() is None


def test_default_does_not_replace_native_functions():
    model = torch.nn.Module()
    model.routed_experts_router = Router()
    method = model.routed_experts_router.forward
    router_objective.install(model, CoreConfig())
    assert model.routed_experts_router.forward == method
    assert model.routed_experts_router.compute_aux_loss().item() == 42.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(router_aux_loss_grouping="bad"),
        dict(router_aux_loss_reduction="bad"),
        dict(router_z_loss_reduction="bad"),
        dict(router_aux_loss_grouping="sequence", compile_model=True),
    ],
)
def test_reject_unsupported_config(kwargs):
    with pytest.raises(ValueError):
        CoreConfig(**kwargs)


def test_context_restored_after_exception():
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        router_objective.batch_context(dict(tokens=torch.zeros(1, 8), total_lengths=[2, 6])),
    ):
        raise RuntimeError("sentinel")
    assert router_objective._CURRENT.get() is None


def test_reject_inconsistent_document_partition():
    with (
        pytest.raises(ValueError, match="partition"),
        router_objective.batch_context(dict(tokens=torch.zeros(1, 8), total_lengths=[2, 5])),
    ):
        pass


def test_reject_dense_model():
    with pytest.raises(ValueError, match="MoE"):
        router_objective.install(
            torch.nn.Linear(2, 2),
            SimpleNamespace(
                router_aux_loss_grouping="sequence",
                router_aux_loss_reduction="token",
                router_z_loss_reduction="token",
                router_aux_count_source="dispatch",
            ),
        )


def test_router_metadata_survives_empty_autograd_context_and_restores():
    model = torch.nn.Module()
    model.routed_experts_router = Router()
    router_objective.install(model, CoreConfig(router_aux_loss_grouping="sequence"))
    router = model.routed_experts_router
    batch = dict(tokens=torch.zeros(1, 8), total_lengths=[2, 6], aux_loss_response_div_factor=2)
    x = torch.randn(1, 8, 2)
    with pytest.raises(RuntimeError, match="sentinel"), router_objective.batch_context(batch, model):
        result = contextvars.Context().run(router, x)
        assert result[3][-1].lengths == (2, 6)
        loss = router.compute_aux_loss(*result[3])
        loss.backward()
        assert torch.isfinite(router.weight.grad).all()
        raise RuntimeError("sentinel")
    assert router._miles_batch is None
    with pytest.raises(RuntimeError, match="batch context"):
        contextvars.Context().run(router, x)


@pytest.mark.parametrize("grouping", ["pack", "sequence"])
@pytest.mark.parametrize("reduction", ["token", "response"])
@pytest.mark.parametrize("z_reduction", ["token", "response"])
@pytest.mark.parametrize("source", ["dispatch", "current"])
@pytest.mark.parametrize("recompute", [False, True])
def test_native_router_combined_controls_match_loss_and_gradient_reference(
    grouping, reduction, z_reduction, source, recompute
):
    model = torch.nn.Module()
    model.routed_experts_router = core_router.MoERouterConfigV2(
        d_model=3,
        num_experts=3,
        top_k=1,
        dtype=core_config.DType.float32,
        lb_loss_weight=0.1,
        z_loss_weight=0.02,
        lb_loss_count_source=source,
    ).build()
    router = model.routed_experts_router
    with torch.no_grad():
        router.weight.copy_(torch.eye(3).reshape_as(router.weight))
    router.replay_expert_indices = torch.full((1, 8, 1), 2)
    options = CoreConfig(
        router_aux_loss_grouping=grouping,
        router_aux_loss_reduction=reduction,
        router_z_loss_reduction=z_reduction,
        router_aux_count_source=source,
    )
    native_forward = router.forward
    router_objective.install(model, options)
    if (grouping, reduction, z_reduction) == ("pack", "token", "token"):
        assert router.forward == native_forward
    x = torch.tensor([[[3.0, 1.0, 0.0]] * 2 + [[1.0, 3.0, 0.0]] * 6], requires_grad=True)
    batch = dict(tokens=torch.zeros(1, 8), total_lengths=[2, 6], aux_loss_response_div_factor=2)

    def forward(value):
        weights, ids, counts, info = router(value, False, loss_div_factor=8.0)
        return weights, ids, counts, router.compute_aux_loss(*info, accumulate_metrics=False)

    with router_objective.batch_context(batch, model):
        weights, ids, counts, loss = (
            checkpoint.checkpoint(forward, x, use_reentrant=False) if recompute else forward(x)
        )
        logits = x @ router.weight.view(3, 3).T
        expected_ids = logits.detach().argmax(-1, keepdim=True) if source == "current" else ids
        lb, z = reference(logits, expected_ids, (2, 6), 8.0, 2.0, grouping, reduction, z_reduction)
        expected = 0.1 * lb + 0.02 * z
        torch.testing.assert_close(loss, expected)
        actual_grad = torch.autograd.grad(loss, (x, router.weight), retain_graph=True)
        reference_grad = torch.autograd.grad(expected, (x, router.weight), retain_graph=True)
        for actual, wanted in zip(actual_grad, reference_grad, strict=True):
            torch.testing.assert_close(actual, wanted)
        # Fresh counts must not alter dispatch, selected weights or their policy gradient.
        assert torch.equal(ids, torch.full_like(ids, 2))
        torch.testing.assert_close(counts, torch.tensor([0, 0, 8], dtype=counts.dtype))
        expected_weights = logits.softmax(-1).gather(-1, ids)
        torch.testing.assert_close(weights, expected_weights)
        actual_policy_grad = torch.autograd.grad(weights.square().sum(), (x, router.weight), retain_graph=True)
        expected_policy_grad = torch.autograd.grad(expected_weights.square().sum(), (x, router.weight))
        for actual, wanted in zip(actual_policy_grad, expected_policy_grad, strict=True):
            torch.testing.assert_close(actual, wanted)


def test_current_counts_reject_dense_model():
    with pytest.raises(ValueError, match="MoE"):
        router_objective.install(torch.nn.Linear(2, 2), CoreConfig(router_aux_count_source="current"))


@pytest.mark.parametrize("aux_enabled", [False, True])
def test_native_dispatch_counters_exclude_scoring_and_recomputation(aux_enabled):
    model = torch.nn.Module()
    model.routed_experts_router = core_router.MoERouterConfigV2(
        d_model=3,
        num_experts=3,
        top_k=1,
        dtype=core_config.DType.float32,
        lb_loss_weight=0.1 if aux_enabled else None,
        z_loss_weight=0.02 if aux_enabled else None,
    ).build()
    router = model.routed_experts_router
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    counts = torch.tensor([3, 1, 0])
    info = (logits.softmax(-1), logits, counts, counts.unsqueeze(0), 4.0)
    contract.auxiliary_metrics(model, reset=True)
    with torch.no_grad():
        router.compute_aux_loss(*info)
    for _ in range(2):
        router.compute_aux_loss(*info)
        router.compute_aux_loss(*info, accumulate_metrics=False)
    result = router_load.collect(router_load.snapshot(model), ep_degree=1)
    assert result["layers"]["routed_experts_router"]["assignments"] == 8
    assert result["summary"]["moe/max_expert_load"] == 6
    assert result["summary"]["moe/dead_experts"] == 1
    contract.auxiliary_metrics(model, reset=True)
    assert router.batch_size_per_expert.count_nonzero() == 0
