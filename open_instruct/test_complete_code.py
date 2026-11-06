"""Complete-test integration preserves failures, source identity and every test."""

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import complete_code

CONFIG = complete_code.Config(api_url="https://service.example", expected_source_sha256="a" * 64)


def mock_client(monkeypatch, scores):
    calls = []

    class Client:
        def __init__(self, config):
            assert config.pass_rate_reward_threshold == 1
            assert config.expected_source_sha256 == CONFIG.expected_source_sha256
            assert config.fresh_process_per_test is True

        async def async_call(self, tokens, prediction, tests):
            calls.append((prediction, tests))
            score = next(scores)
            if isinstance(score, Exception):
                raise score
            return SimpleNamespace(score=score, diagnostics={"tests": len(tests)})

    monkeypatch.setattr(
        complete_code.importlib,
        "import_module",
        lambda name: SimpleNamespace(CodeVerifier=Client, ServiceConfig=SimpleNamespace),
    )
    return calls


def test_complete_suite_keeps_tests_after_failure_and_beyond_client_limit(monkeypatch):
    calls = mock_client(monkeypatch, iter([0, 1]))
    tests = [f"assert f({i}) == {i}" for i in range(2001)]
    result = asyncio.run(
        complete_code.CompleteCodeVerifier(CONFIG).async_call(
            [], "<think>reasoning</think>```python\ndef f(x): return x\n```", {"mode": "assert", "tests": tests}
        )
    )
    assert result.score == 0
    assert [test for _, chunk in calls for test in chunk] == tests
    assert all("reasoning" not in response for response, _ in calls)
    assert result.diagnostics["tests"] == 2001
    assert result.diagnostics["service_source_sha256"] == "a" * 64


def test_unknown_service_failure_is_not_zero(monkeypatch):
    mock_client(monkeypatch, iter([RuntimeError("incomplete execution")]))
    with pytest.raises(RuntimeError, match="incomplete execution"):
        asyncio.run(
            complete_code.CompleteCodeVerifier(CONFIG).async_call([], "pass", {"mode": "assert", "tests": ["pass"]})
        )


@pytest.mark.parametrize(
    "target",
    [
        [],
        {"mode": "assert", "tests": []},
        {"mode": "assert", "tests": [3]},
        {"mode": "stdio", "tests": [{"input": "x", "output": 2}]},
    ],
)
def test_malformed_targets_fail_before_client(target):
    with pytest.raises(ValueError):
        asyncio.run(complete_code.CompleteCodeVerifier(CONFIG).async_call([], "pass", target))


@pytest.mark.parametrize(
    "change",
    [
        {"expected_source_sha256": ""},
        {"api_url": "https://user:secret@example.com"},
        {"max_execution_time": 0},
        {"total_timeout": float("inf")},
    ],
)
def test_configuration_errors_are_fatal(change):
    with pytest.raises(ValueError):
        complete_code.CompleteCodeVerifier(dataclasses.replace(CONFIG, **change))


def test_acecode_preserves_context_and_all_assertions():
    context = [{"role": "system", "content": "Use Python"}, {"role": "user", "content": "Implement f"}]
    raw = {"context_messages": context, "question": "Implement f", "test_cases": ["assert f() == 1"] * 2001}
    messages, label, name, stratum = sources.adapt(raw, "acecode_v1")
    assert messages == context and messages is not context
    assert label["tests"] == raw["test_cases"] and len(label["tests"]) == 2001
    assert name == "complete_code" and stratum == "acecode_v1:assert"


def test_livecodebench_adaptation_preserves_test_order():
    pytest.importorskip("code_execution_client.livecodebench")
    raw = {"question": "Add", "verifier_metadata": {"unit_tests": {"inputs": ["2 3", "4 5"], "outputs": ["5", "9"]}}}
    _, label, name, stratum = sources.adapt(raw, "livecodebench_v1")
    assert label == {"mode": "stdio", "tests": [{"input": "2 3", "output": "5"}, {"input": "4 5", "output": "9"}]}
    assert name == "complete_code" and stratum == "livecodebench_v1:stdio"


def test_invalid_source_assertion_is_a_target_error():
    with pytest.raises(ValueError, match="source test"):
        complete_code.validate_target({"mode": "assert", "tests": ["assert f( == 1"]})
