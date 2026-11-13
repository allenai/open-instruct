"""Native extraction, scoring transport, and data-binding controls for C033."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.rewards import abstention, general_judge, judge_registry, judging

TARGET = {"kind": abstention.KIND, "question": "What is the capital of France?", "answer": "Paris"}


@pytest.fixture
def binding(monkeypatch):
    document = {
        "judges": {"qa": {"mode": "external", "model": "judge", "endpoint": "http://judge.invalid/v1"}},
        "rubrics": {"qa": {"profile": abstention.PROFILE, "temperature": 0, "max_response_tokens": 64}},
        "judging": {"bindings": {"abstention": {"judge": "qa", "rubric": "qa"}}},
    }
    value = judging.registry(judging.parse(document))
    monkeypatch.setenv(judging.REGISTRY_ENV, json.dumps(value))

    async def no_delay(*args):
        pass

    monkeypatch.setattr(general_judge.asyncio, "sleep", no_delay)
    return value


def score(response):
    sample = SimpleNamespace(response=response, metadata={"judge_query": TARGET["question"]})
    reward = asyncio.run(
        general_judge.general_judge_score(SimpleNamespace(), sample, name="abstention", target=TARGET)
    )
    return reward, sample.metadata["verifier_diagnostics"]["abstention"]


@pytest.mark.parametrize(
    "response", [r"\boxed{[IDK]}", "[idk]", r"<think>Paris</think>\boxed{[IDK]}", r"reasoning</think>\boxed{[IDK]}"]
)
def test_explicit_abstention_never_calls_judge(monkeypatch, binding, response):
    request = Mock(side_effect=AssertionError("Explicit abstention requires no judge call"))
    monkeypatch.setattr(general_judge, "_request", request)
    reward, diagnostic = score(response)
    assert reward == 0.5 and diagnostic["verdict"] == "abstain"
    assert diagnostic["judge_called"] is False
    request.assert_not_called()


@pytest.mark.parametrize(
    "reply,expected,verdict", [("A", 1.0, "correct"), ("B", 0.0, "incorrect"), ("C", 0.5, "abstain")]
)
def test_native_judge_rewards(monkeypatch, binding, reply, expected, verdict):
    request = Mock(return_value=reply)
    monkeypatch.setattr(general_judge, "_request", request)
    reward, diagnostic = score(r"\boxed{Paris}")
    assert reward == expected and diagnostic["verdict"] == verdict
    assert diagnostic["judge_called"] is True
    prompt = request.call_args.args[1]
    assert "Predicted answer: Paris" in prompt
    assert "Gold target: Paris" in prompt


@pytest.mark.parametrize("reply", ["", "<think>reasoning</think>"])
def test_empty_judge_is_unknown_even_with_default_zero_policy(monkeypatch, binding, reply):
    monkeypatch.setattr(general_judge, "_request", lambda *args: reply)
    with pytest.raises(general_judge.JudgeResponseError, match="empty"):
        score("Paris")


def test_unavailable_judge_is_unknown(monkeypatch, binding):
    def unavailable(*args):
        raise general_judge.JudgeResponseError("service unavailable")

    monkeypatch.setattr(general_judge, "_request", unavailable)
    with pytest.raises(general_judge.JudgeResponseError, match="service unavailable"):
        score("Paris")


def test_nested_braces_and_last_box_match_native_extraction():
    assert abstention.extracted_answer(r"\boxed{old} then \boxed{a{b}c}") == "a{b}c"
    assert abstention.extracted_answer(r"\boxed{unfinished") == r"\boxed{unfinished"
    assert not abstention.is_abstention(r"<think>\boxed{[IDK]}</think>Paris")
    # The native profile permits unboxed answers and uses its original permissive
    # A/B/C parser. These are pinned semantics, not strict-format enforcement.
    assert abstention.grade("unparseable lowercase reply") == ("incorrect", 0.0)


def test_abstention_target_cannot_use_generic_rubric(monkeypatch, binding):
    binding["rubrics"]["qa"]["profile"] = "open-instruct/general-quality_ref"
    monkeypatch.setenv(judging.REGISTRY_ENV, json.dumps(binding))
    with pytest.raises(ValueError, match="require.*rubric"):
        judge_registry.resolve_request(SimpleNamespace(), "abstention", "question", "Paris", TARGET)


def test_target_and_prompt_validation():
    for target in ("Paris", {**TARGET, "answer": ""}, {**TARGET, "extra": 1}):
        with pytest.raises(ValueError):
            abstention.build_prompt("Paris", target)
    prompt, digest = abstention.build_prompt("Paris", TARGET)
    assert TARGET["question"] in prompt
    assert digest == hashlib.sha256(abstention._DEFAULT_JUDGE_PROMPT_TEMPLATE.encode()).hexdigest()


def test_pinned_rubric_drift_fails(monkeypatch):
    monkeypatch.setattr(abstention, "_DEFAULT_JUDGE_PROMPT_TEMPLATE", "changed")
    with pytest.raises(RuntimeError, match="rubric drift"):
        abstention.build_prompt("Paris", TARGET)
