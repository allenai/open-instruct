"""Final-answer dispatch must not reapply a verifier's raw-response boundary gate."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import code_rewards, rewards


def sample(response, status="completed", label="computer", verifier="ifeval"):
    return SimpleNamespace(
        response=response,
        tokens=[1, 2, 3],
        response_length=2,
        status=status,
        prompt="Include computer.",
        metadata={"verifiers": [{"name": verifier, "target": label}]},
    )


def score(registry, item, final_only):
    args = SimpleNamespace(
        olmo_core=CoreConfig(
            reward_config=str(registry), reward_final_answer_only=final_only, reward_zero_truncated=True
        )
    )
    return asyncio.run(rewards.registered_reward(args, item))


@pytest.mark.parametrize("final_only", [False, True])
def test_code_verifier_keeps_normal_method_dispatch(tmp_path, monkeypatch, final_only):
    registry = tmp_path / "code.json"
    registry.write_text(
        json.dumps(
            {
                "code": {
                    "factory": "open_instruct.miles.rewards.code_rewards.CodeVerifier",
                    "config": {"api_url": "https://unused.invalid"},
                }
            }
        )
    )
    observed = []

    async def execute(args, prediction, label, **kwargs):
        observed.append((prediction, label))
        return 1.0, {"status": "completed"}

    monkeypatch.setattr(code_rewards, "execute", execute)
    response = "scratch </think>draft </think>print(42)"
    item = sample(response, label=["assert True"], verifier="code")
    assert score(registry, item, final_only) == 1.0
    assert observed == [("print(42)" if final_only else response, ["assert True"])]
    assert item.response == response and item.tokens == [1, 2, 3]


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
