"""Analytic and lifecycle checks for the isolated half-group probe."""

import itertools
import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles.diagnostics import gradient_noise as probe


def test_partition_keeps_siblings_across_ranks():
    ranks = [list(range(0, 32, 2)), list(range(1, 32, 2))]
    selected, groups = probe.partition(ranks, 8, 17)
    assert groups == 4
    assert len(selected) == 2
    assert sum(i // 8 in selected for rank in ranks for i in rank) == 16
    assert probe.partition(ranks, 8, 17) == (selected, groups)


@pytest.mark.parametrize("indices", [[[0, 1, 2]], [[0, 1, 2, 2]], [[0, 1, 4, 5, 8, 9]]])
def test_partition_rejects_missing_duplicate_and_odd_groups(indices):
    with pytest.raises(ValueError):
        probe.partition(indices, 2, 17)


def test_statistics_matches_two_independent_half_gradients():
    a = torch.tensor([1.0, 3.0], dtype=torch.float64)
    b = torch.tensor([3.0, -1.0], dtype=torch.float64)
    full = (a + b) / 2
    row = probe.statistics(float(a @ a), float(full @ full), float(a @ full), 64, 8)
    assert row["signal_power"] == float(a @ b) == 0
    assert row["full_batch_noise_power"] == float((a - b).square().sum() / 4) == 5
    assert row["noise_scale_responses"] is None
    row = probe.statistics(1, 0, 0, 64, 8)
    assert row["signal_power"] == -1
    assert row["noise_scale_responses"] is None


def test_noise_estimator_over_all_independent_half_draws():
    # Population gradient mean=2, variance=1. Each half has one group.
    signal, noise = [], []
    for a, b in itertools.product([1.0, 3.0], repeat=2):
        full = (a + b) / 2
        row = probe.statistics(a * a, full * full, a * full, 2, 8)
        signal.append(row["signal_power"])
        noise.append(row["full_batch_noise_power"])
    assert sum(signal) / 4 == 4
    assert sum(noise) / 4 == 0.5
    assert 2 * sum(noise) / sum(signal) == 0.25  # population variance / mean squared


def test_masked_extra_backward_preserves_full_gradient(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    batches = [
        {
            "sample_indices": list(range(16)),
            "advantages": [torch.tensor(float(i % 3 - 1)) for i in range(16)],
            "features": torch.arange(32, dtype=torch.float32).reshape(16, 2) / 10,
        }
    ]
    calls = []

    def train_batch_with_loss(items, objective, context):
        calls.append(1)
        results = []
        for batch in items:
            loss, values = objective(None, batch)
            loss.backward()
            results.append(values)
        return results

    module = SimpleNamespace(
        train_batch_with_loss=train_batch_with_loss, zero_grads=lambda: setattr(parameter, "grad", None)
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(
            calculate_per_token_loss=False,
            entropy_coef=0,
            kl_coef=0,
            use_kl_loss=False,
            olmo_core=SimpleNamespace(router_aux_loss_weight=0, router_z_loss_weight=0),
            n_samples_per_prompt=8,
            global_batch_size=16,
            save=str(tmp_path / "checkpoints"),
        ),
        train_module=module,
        optimizer=None,
        clock=SimpleNamespace(completed_steps=0),
    )

    def objective(module, batch):
        scores = batch["features"] @ parameter
        return -(scores * torch.stack(batch["advantages"])).mean(), {}

    objective(None, batches[0])[0].backward()
    expected = parameter.grad.clone()
    parameter.grad = None
    originals = batches[0]["advantages"]
    monkeypatch.setattr(probe, "gradient_slices", lambda _: iter([("weight", parameter.grad, 1.0, 1)]))
    monkeypatch.setattr(probe.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(probe.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(probe.dist, "all_reduce", lambda _: None)
    monkeypatch.setattr(probe.dist, "all_gather_object", lambda output, value: output.__setitem__(0, value))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _: None)
    probe.measure(actor, batches, objective, None)
    torch.testing.assert_close(parameter.grad, expected)
    assert batches[0]["advantages"] is originals
    assert len(calls) == 2
    row = json.loads((tmp_path / "gradient-noise.jsonl").read_text())
    assert row["responses"] == 16
    assert row["parameter_elements"] == 2
    assert row["full_norm2"] == pytest.approx(float(expected.square().sum()), rel=1e-6)


def test_compare_applies_expert_scale_and_replica_weight(monkeypatch):
    first = {"replicated": (torch.tensor([2.0, 4.0]), 1.0, 2), "expert": (torch.tensor([6.0]), 0.5, 1)}
    values = [("replicated", torch.tensor([4.0, 2.0]), 1.0, 2), ("expert", torch.tensor([2.0]), 0.5, 1)]
    monkeypatch.setattr(probe, "gradient_slices", lambda _: iter(values))
    monkeypatch.setattr(probe.dist, "all_reduce", lambda _: None)
    # Replicated entries count half on each rank; experts get EP scaling squared.
    assert probe.compare(None, first) == [19.0, 11.0, 11.0, 2.0]
