"""Arithmetic and pinned native integration checks; no GPU allocation needed."""

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from scripts.miles.bernoulli import install

from open_instruct.miles.rewards import bernoulli


@pytest.mark.parametrize("n", [2, 3, 8, 16])
def test_every_binary_success_count(n):
    for k in range(n + 1):
        rewards = torch.tensor([1.0] * k + [0.0] * (n - k))
        torch.testing.assert_close(bernoulli.reference_std(rewards), rewards.std())


def test_partial_rewards_preserve_distance():
    rewards = torch.tensor([0.3] * 4 + [0.4] * 4)
    denominator = bernoulli.reference_std(rewards)
    assert denominator.item() == pytest.approx(0.50990195)
    advantage = (rewards - rewards.mean()) / (denominator + 1e-6)
    assert advantage[-1].item() == pytest.approx(0.098058, abs=1e-6)
    narrow = torch.tensor([0.349] * 4 + [0.351] * 4)
    assert bernoulli.reference_std(narrow).item() == pytest.approx(denominator.item())
    assert (narrow[-1] - narrow.mean()).item() == pytest.approx(0.001, abs=1e-7)


@pytest.mark.parametrize("values", [[0.0] * 8, [1.0] * 8, [0.3] * 8, [0.4]])
def test_constant_groups(values):
    rewards = torch.tensor(values)
    advantage = (rewards - rewards.mean()) / (bernoulli.reference_std(rewards) + 1e-6)
    assert torch.isfinite(advantage).all()
    torch.testing.assert_close(advantage, torch.zeros_like(rewards), atol=1e-7, rtol=0)


@pytest.mark.parametrize("values", [[-0.1, 1.0], [0.0, 1.1], [float("nan"), 0], [float("inf"), 0], []])
def test_invalid_rewards_fail(values):
    with pytest.raises(ValueError):
        bernoulli.reference_std(torch.tensor(values))


def native_namespace(patched):
    source = Path(os.environ["OI_BERNOULLI_NATIVE_SOURCE"]).read_text()
    if patched:
        source = install.patched_source(source, hashlib.sha256(source.encode()).hexdigest())
    names = {"_reward_group_segments", "_normalize_rewards_by_rollout", "_post_process_rewards"}
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "Any": Any, "Sample": SimpleNamespace, "bernoulli": bernoulli}
    exec(compile(tree, "pinned_native", "exec"), namespace)
    return namespace


def test_native_groups_raw_rewards_and_opt_out():
    old, new = native_namespace(False), native_namespace(True)
    rewards = [1.0] + [0.0] * 7 + [0.3] * 4 + [0.4] * 4
    samples = [
        SimpleNamespace(group_index=i // 8, rollout_id=i, index=i, get_reward_value=lambda args, r=r: r)
        for i, r in enumerate(rewards)
    ]
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=8,
        rollout_batch_size=2,
    )
    raw, advantages = new["_post_process_rewards"](args, samples, None)
    original = old["_post_process_rewards"](args, samples, None)[1]
    assert raw == rewards
    assert advantages[:8] == pytest.approx(original[:8])
    assert abs(advantages[8]) < abs(original[8]) / 9
    # Interleaving must not mix prompt groups.
    order = [i for pair in zip(range(8), range(8, 16)) for i in pair]
    shuffled = new["_post_process_rewards"](args, [samples[i] for i in order], None)[1]
    assert shuffled == pytest.approx([advantages[i] for i in order])
    args.grpo_std_normalization = False
    assert new["_post_process_rewards"](args, samples, None) == old["_post_process_rewards"](args, samples, None)
    args.rewards_normalization = False
    assert new["_post_process_rewards"](args, samples, None) == (rewards, rewards)


def test_installer_rejects_unknown_native():
    with pytest.raises(ValueError, match="pinned"):
        install.patched_source("unexpected", "wrong")
