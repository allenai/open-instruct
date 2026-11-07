"""Final-answer dispatch must not reapply a verifier's raw-response boundary gate."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


def sample(response, status="completed", label="computer", verifier="ifeval"):
    return SimpleNamespace(
        response=response,
        tokens=[1, 2, 3],
        response_length=2,
        status=status,
        prompt="Include computer.",
        metadata={"verifiers": [{"name": verifier, "target": label}]},
    )


def score(registry, item, final_only, reward_function=rewards.registered_reward):
    args = SimpleNamespace(
        olmo_core=CoreConfig(
            reward_config=str(registry), reward_final_answer_only=final_only, reward_zero_truncated=True
        )
    )
    return asyncio.run(reward_function(args, item))


@pytest.fixture
def legacy_verifier(tmp_path):
    registry = tmp_path / "legacy.json"
    registry.write_text(
        json.dumps({"legacy": {"factory": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier"}})
    )
    return registry


@pytest.mark.parametrize("reward_function", [rewards.registered_reward, rewards.rollout_reward])
@pytest.mark.parametrize(
    "response,status",
    [
        ("reasoning </think><answer>42</answer>", "completed"),
        ("<answer>42</answer>", "completed"),
        ("reasoning </think> ", "completed"),
        ("reasoning </think><answer>42</answer>", "truncated"),
    ],
)
def test_final_only_rejects_unsupported_verifier_even_when_gated(legacy_verifier, response, status, reward_function):
    item = sample(response, status=status, label="42", verifier="legacy")
    item.group_index = 7
    with pytest.raises(ValueError, match="Verifier 'legacy' must implement async_call_final_answer"):
        score(legacy_verifier, item, True, reward_function)
    assert "reward_omission" not in item.metadata
    assert "reward_gate" not in item.metadata
    assert item.response == response and item.tokens == [1, 2, 3]


def test_raw_mode_keeps_legacy_verifier_support(legacy_verifier):
    assert score(legacy_verifier, sample("<answer>42</answer>", label="42", verifier="legacy"), False) == 1


def test_final_only_rejects_noncallable_entry_point(legacy_verifier, monkeypatch):
    verifier = rewards._registry(str(legacy_verifier))["legacy"]
    monkeypatch.setattr(verifier, "async_call_final_answer", None, raising=False)
    with pytest.raises(ValueError, match="must implement async_call_final_answer"):
        score(legacy_verifier, sample("reasoning </think><answer>42</answer>", verifier="legacy"), True)


@pytest.fixture
def r1_format(tmp_path):
    registry = tmp_path / "r1-format.json"
    registry.write_text(
        json.dumps({"r1_format": {"factory": "open_instruct.miles.rewards.task_verifiers.R1FormatVerifier"}})
    )
    return registry


@pytest.mark.parametrize("final_only", [False, True])
@pytest.mark.parametrize(
    "response,raw_expected,final_expected",
    [
        ("reasoning </think><answer>42</answer>", 1, 1),
        ("reasoning </think> \n\t<answer>line one\nline two</answer>", 1, 1),
        ("reasoning </think><answer>42</answer> trailing text", 1, 1),
        # This is the existing format check, not an answer-content validator.
        ("reasoning </think><answer></answer>", 1, 1),
        ("reasoning </think>42</answer>", 0, 0),
        ("reasoning </think><answer>42", 0, 0),
        ("reasoning </think>prefix <answer>42</answer>", 0, 0),
        ("<answer>42</answer> </think>wrong", 0, 0),
        ("draft </think><answer>42</answer> </think>wrong", 1, 0),
        ("draft </think>wrong </think><answer>42</answer>", 1, 1),
    ],
)
def test_r1_format_checks_selected_region_and_preserves_raw(
    r1_format, final_only, response, raw_expected, final_expected
):
    item = sample(response, verifier="r1_format")
    assert score(r1_format, item, final_only) == (final_expected if final_only else raw_expected)
    assert item.response == response and item.tokens == [1, 2, 3]
    assert item.metadata["reward_components"] == [
        {"name": "r1_format", "score": final_expected if final_only else raw_expected, "weight": 1.0, "cost": 0.0}
    ]


@pytest.mark.parametrize("response", ["<answer>42</answer>", "reasoning </think>", "reasoning </think> \n\t"])
def test_r1_format_cannot_bypass_missing_or_empty_final_gate(r1_format, response):
    item = sample(response, verifier="r1_format")
    assert score(r1_format, item, True) == 0
    assert item.metadata["reward_gate"] == "missing_final_answer"


@pytest.mark.parametrize("final_only", [False, True])
def test_r1_format_cannot_bypass_truncation_gate(r1_format, final_only):
    item = sample("reasoning </think><answer>42</answer>", "truncated", verifier="r1_format")
    assert score(r1_format, item, final_only) == 0
    assert item.metadata["reward_gate"] == "truncated"


@pytest.mark.parametrize("final_only", [False, True])
def test_r1_format_retains_group_reward_contrast(r1_format, final_only):
    items = [
        sample("reasoning </think>" + answer, verifier="r1_format")
        for answer in ["<answer>42</answer>", "42", "<answer>wrong</answer>", "<answer>42"]
    ]
    assert score(r1_format, items, final_only) == [1.0, 0.0, 1.0, 0.0]


def test_final_dispatch_receives_suffix_without_reasoning_tokens(r1_format, monkeypatch):
    verifier = rewards._registry(str(r1_format))["r1_format"]
    original = verifier.async_call_final_answer
    observed = []

    async def record(tokens, prediction, label, **kwargs):
        observed.append((tokens, prediction, kwargs))
        label["mutable"].append("verifier mutation")
        return await original(tokens, prediction, label, **kwargs)

    monkeypatch.setattr(verifier, "async_call_final_answer", record)
    item = sample("scratch </think>draft </think><answer>42</answer>", label={"mutable": []}, verifier="r1_format")
    assert score(r1_format, item, True) == 1
    assert observed == [([], "<answer>42</answer>", {"query": "Include computer.", "rollout_state": None})]
    assert item.metadata["verifiers"][0]["target"] == {"mutable": []}
    assert item.response == "scratch </think>draft </think><answer>42</answer>"
    assert item.tokens == [1, 2, 3]
