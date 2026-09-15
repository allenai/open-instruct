"""Malformed judge outputs cannot terminate a run or masquerade as valid grades."""

import asyncio
from types import SimpleNamespace

import pytest
import requests

from open_instruct.miles import general_judge


@pytest.mark.parametrize("reply", ["**SCORE:** 1", "**SCORE**: 1", "SCORE: 1", '{"REASONING":"poor", "SCORE":1}'])
def test_explicit_score_formats_agree(reply):
    assert general_judge.parse_judge_response(reply)[1] == 0.1


@pytest.mark.parametrize("reply", ["I considered 1 and 10.", "SCORE: 1\nSCORE: 9", "**SCORE:** 11", '{"SCORE": true}'])
def test_ambiguous_and_invalid_scores_remain_rejected(reply):
    with pytest.raises(general_judge.JudgeResponseError):
        general_judge.parse_judge_response(reply)


@pytest.fixture
def request_setup(monkeypatch):
    monkeypatch.setenv("OI_MILES_JUDGE_API_BASE", "http://judge.invalid/v1")
    monkeypatch.delenv("OI_MILES_JUDGE_FAILURE_POLICY", raising=False)
    monkeypatch.setattr(general_judge.judge_registry, "resolve_request", lambda *a: None)

    async def no_delay(*args):
        pass

    monkeypatch.setattr(general_judge.asyncio, "sleep", no_delay)
    return SimpleNamespace(response="candidate", metadata={"judge_query": "query"})


def run_score(sample, policy=None):
    return asyncio.run(
        general_judge.general_judge_score(
            SimpleNamespace(llm_judge_failure_policy=policy), sample, name="general-quality", target=None
        )
    )


def test_exhausted_malformed_reply_is_measured_zero_then_next_request_can_succeed(monkeypatch, request_setup):
    replies = iter(["malformed"] * 3 + ['{"SCORE": 8}'])
    monkeypatch.setattr(general_judge, "_request", lambda *a: next(replies))
    assert run_score(request_setup) == 0
    diagnostics = request_setup.metadata["verifier_diagnostics"]["general-quality"]
    assert diagnostics["status"] == "judge_error"
    assert len(diagnostics["attempts"]) == 3
    assert diagnostics["attempts"][0]["raw_reply"] == "malformed"
    assert run_score(request_setup) == 0.8
    assert request_setup.metadata["verifier_diagnostics"]["general-quality"]["status"] == "ok"


@pytest.mark.parametrize("policy", ["raise", "invalid"])
def test_strict_or_invalid_policy_does_not_return_a_grade(monkeypatch, request_setup, policy):
    monkeypatch.setattr(general_judge, "_request", lambda *a: "malformed")
    with pytest.raises((general_judge.JudgeResponseError, ValueError)):
        run_score(request_setup, policy)


def test_context_or_programming_errors_do_not_become_zero(monkeypatch, request_setup):
    def fail(*args):
        raise RuntimeError("judge context overflow")

    monkeypatch.setattr(general_judge, "_request", fail)
    with pytest.raises(RuntimeError, match="context overflow"):
        run_score(request_setup)


def test_request_timeout_is_classified_as_recoverable(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(general_judge, "_get_session", lambda: SimpleNamespace(post=fail))
    config = general_judge.GeneralJudgeConfig(
        api_url="http://judge.invalid/v1/chat/completions",
        api_key="EMPTY",
        model="model",
        max_tokens=10,
        max_context_length=100,
        temperature=1,
        timeout=1,
        seed=1,
        max_concurrent_calls=1,
    )
    with pytest.raises(general_judge.JudgeResponseError, match="Timeout"):
        general_judge._request(config, "prompt")
