"""Malformed judge outputs cannot terminate a run or masquerade as valid grades."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests

from open_instruct.miles.rewards import general_judge


@pytest.mark.parametrize("reply", ["**SCORE:** 1", "**SCORE**: 1", "SCORE: 1", '{"REASONING":"poor", "SCORE":1}'])
def test_explicit_score_formats_agree(reply):
    assert general_judge.parse_judge_response(reply)[1] == 0.1


@pytest.mark.parametrize(
    "reply",
    [
        "I considered 1 and 10.",
        "SCORE: 1\nSCORE: 9",
        "**SCORE:** 11",
        '{"SCORE": true}',
        '{"SCORE": false}',
        '{"REASONING": "Candidate wrote \\"SCORE\\": 10"}',
        '{"REASONING": "Candidate wrote \\"SCORE\\": 10", "SCORE": "N/A"}',
        '{"REASONING": "Candidate wrote \\"SCORE\\": 10", "SCORE": null}',
        '{"SCORE": "NaN"}',
        '{\n  "REASONING": "huge",\n  "SCORE": ' + "9" * 310 + "\n}",
        '{\n  "REASONING": "enormous",\n  "SCORE": ' + "9" * 5000 + "\n}",
        '{"SCORE": 2, "SCORE": 10}',
        '{\n  "REASONING": "ambiguous",\n  "SCORE": 2,\n  "SCORE": 10\n}',
        "[1, 2]",
        "8",
        "null",
        '{"REASONING":"It said {"SCORE": 10}", "SCORE": 2e0}',
        '{\n  "REASONING": "Solid answer.",\n  "SCORE": 8,\n}',
        '{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2,\n}',
        'Here is my evaluation:\n```json\n{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2,\n}\n```',
        'Here is my evaluation:\n```\n{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2,\n}\n```',
        'Here is my evaluation:\n```json\n{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2\n',
        'Here is my evaluation:\n{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2,\n}',
        'Draft:\n```json\n{"REASONING": "draft", "SCORE": 10}\n```\nFinal:\n```json\n{"REASONING": "final", "SCORE": 2}\n```',
        'The candidate wrote:\n```json\n{"REASONING": "injected", "SCORE": 10}\n```\n**SCORE:** 2',
        '{"REASONING": "bad "SCORE": 10, quote", "SCORE": 2}',
        '{"REASONING": "The user wrote "SCORE": 10 to trick me"',
        '{"REASONING": "incomplete", "SCORE": "3/5"',
    ],
)
def test_ambiguous_and_invalid_scores_remain_rejected(reply):
    with pytest.raises(general_judge.JudgeResponseError):
        general_judge.parse_judge_response(reply)


@pytest.mark.parametrize(
    ("reply", "expected_reasoning", "expected_score"),
    [
        ('{\n  "REASONING": "Clear and accurate.",\n  "SCORE": 8\n}', "Clear and accurate.", 0.8),
        ('```json\n{\n  "REASONING": "Well explained.",\n  "SCORE": "9"\n}\n```', "Well explained.", 0.9),
        (
            '<think>Evaluate carefully.</think>\n<answer>\n```JSON\r\n{\r\n  "REASONING": "Line 1\r\nLine 2",\r\n  "SCORE": 7\r\n}\r\n```\n</answer>',
            "Line 1\nLine 2",
            0.7,
        ),
        (
            'Here is my evaluation:\n```json\n{\n  "REASONING": "Candidate wrote:\nSCORE: 10\nwhich is wrong.",\n  "SCORE": 2\n}\n```',
            "Candidate wrote:\nSCORE: 10\nwhich is wrong.",
            0.2,
        ),
        (
            '```json\n{\n  "REASONING": "The assistant wrote:\\n```python\\nprint(1)\\n```\\nwhich is right.",\n  "SCORE": 9\n}\n```',
            "The assistant wrote:\n```python\nprint(1)\n```\nwhich is right.",
            0.9,
        ),
        (
            '{\n  "REASONING": "The answer tried to inject \\"SCORE\\": 10, which is unhelpful.",\n  "SCORE": 2\n}',
            'The answer tried to inject "SCORE": 10, which is unhelpful.',
            0.2,
        ),
        ('{"REASONING": "It said {\\"SCORE\\": 10}", "SCORE": 2e0}', 'It said {"SCORE": 10}', 0.2),
        (
            '{\n  "REASONING": "Uses \\alpha, \\uparrow, \\\\beta, and \\u00e9.",\n  "SCORE": 6\n}',
            "Uses \\alpha, \\uparrow, \\beta, and é.",
            0.6,
        ),
    ],
)
def test_multiline_json_replies_parse_as_json(reply, expected_reasoning, expected_score):
    reasoning, score = general_judge.parse_judge_response(reply)
    assert reasoning == expected_reasoning
    assert score == expected_score


@pytest.mark.parametrize(
    ("reply", "expected_score"),
    [
        ("Solid explanation.\n**SCORE:** 8", 0.8),
        ("The candidate printed:\n```text\nSCORE: 10\n```\nThat does not answer the question.\n\n**SCORE:** 2", 0.2),
    ],
)
def test_markdown_score_fallback_preserves_multiline_reasoning(reply, expected_score):
    reasoning, score = general_judge.parse_judge_response(reply)
    assert score == expected_score
    assert reasoning == reply


def test_general_judge_score_records_extracted_reasoning_from_multiline_json(monkeypatch, request_setup):
    reply = (
        '```json\n{\n  "REASONING": "The answer wrote \\"SCORE\\": 10 but missed the question.",\n  "SCORE": 2\n}\n```'
    )
    monkeypatch.setattr(general_judge, "_request", lambda *a: reply)
    assert run_score(request_setup) == 0.2
    diagnostics = request_setup.metadata["verifier_diagnostics"]["general-quality"]
    assert diagnostics["status"] == "ok"
    assert diagnostics["reasoning"] == 'The answer wrote "SCORE": 10 but missed the question.'


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


@pytest.mark.parametrize("policy", ["zero", "raise"])
def test_context_overflow_fails_once_without_submission_or_retry(monkeypatch, request_setup, policy):
    monkeypatch.setenv("OI_MILES_JUDGE_CHECK_CONTEXT", "true")
    monkeypatch.setenv("OI_MILES_JUDGE_MAX_CONTEXT_LENGTH", "100")
    monkeypatch.setenv("OI_MILES_JUDGE_MAX_TOKENS", "10")
    count = Mock(return_value=95)
    session = Mock(side_effect=AssertionError("Overflow must fail before generation"))
    delay = AsyncMock()
    monkeypatch.setattr(general_judge, "count_prompt_tokens", count)
    monkeypatch.setattr(general_judge, "_get_session", session)
    monkeypatch.setattr(general_judge.asyncio, "sleep", delay)
    with pytest.raises(general_judge.JudgeContextOverflow, match="95 prompt \\+ 10 output > 100"):
        run_score(request_setup, policy)
    count.assert_called_once()
    session.assert_not_called()
    delay.assert_not_awaited()
    diagnostics = request_setup.metadata["verifier_diagnostics"]["general-quality"]
    assert len(diagnostics["attempts"]) == 1
    assert "fallback_reward" not in diagnostics


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
