"""Complete-test integration preserves failures, source identity and every test."""

import asyncio
import dataclasses
import importlib
from types import SimpleNamespace

import pytest
import requests

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import complete_code, rewards

CONFIG = complete_code.Config(api_url="https://service.example", expected_source_sha256="a" * 64)


def mock_client(monkeypatch, scores):
    calls = []
    native_import = importlib.import_module

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
            if isinstance(score, SimpleNamespace):
                return score
            return SimpleNamespace(score=score, diagnostics={"status": "ok", "tests": len(tests)})

    monkeypatch.setattr(
        complete_code.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(CodeVerifier=Client, ServiceConfig=SimpleNamespace)
            if name == "code_execution_client.verifier"
            else native_import(name)
        ),
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


@pytest.mark.parametrize("status", ["service_error", "timeout", "incomplete", "rejected", None])
@pytest.mark.parametrize("training", [True, False])
def test_failed_partition_is_not_a_wrong_answer(monkeypatch, tmp_path, status, training):
    mock_client(monkeypatch, iter([0, SimpleNamespace(score=0.0, diagnostics={"status": status})]))
    verifier = complete_code.CompleteCodeVerifier(CONFIG)
    monkeypatch.setattr(rewards, "_registry", lambda path: {"complete_code": verifier})
    sample = SimpleNamespace(
        group_index=0,
        tokens=[],
        response_length=0,
        response="pass",
        prompt="Write code",
        metadata={"verifiers": [{"name": "complete_code", "target": {"mode": "assert", "tests": ["pass"] * 2001}}]},
    )
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"), save=str(tmp_path))
    if training:
        assert asyncio.run(rewards.rollout_reward(args, sample)) is None
        assert "reward_components" not in sample.metadata
    else:
        with pytest.raises(RuntimeError, match="failed to grade"):
            asyncio.run(rewards.registered_reward(args, sample))
    diagnostics = sample.metadata["verifier_diagnostics"]["complete_code"]
    assert diagnostics["status"] == "service_error"
    assert diagnostics["partitions"][-1]["status"] == status


def test_completed_invalid_code_remains_a_zero_reward(monkeypatch):
    mock_client(
        monkeypatch, iter([SimpleNamespace(score=0, diagnostics={"status": "completed", "outcome": "invalid_code"})])
    )
    result = asyncio.run(
        complete_code.CompleteCodeVerifier(CONFIG).async_call([], "", {"mode": "assert", "tests": ["pass"]})
    )
    assert result.score == 0
    assert result.diagnostics["status"] == "ok"


def test_complete_code_service_metrics_include_partition_failures(monkeypatch):
    metrics = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    mock_client(monkeypatch, iter([1, SimpleNamespace(score=0, diagnostics={"status": "service_error"})]))
    verifier = complete_code.CompleteCodeVerifier(CONFIG)
    results = [asyncio.run(verifier.async_call([], "pass", {"mode": "assert", "tests": ["pass"]})) for _ in range(2)]
    values = metrics.code_service_metrics(
        [SimpleNamespace(metadata={"verifier_diagnostics": {"code": r.diagnostics}}) for r in results]
    )
    assert values["rollout/code_verifier/samples"] == 2
    assert values["rollout/code_verifier/service_errors"] == 1
    assert values["rollout/code_verifier/service_error_fraction"] == 0.5


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
    assert label == {
        "mode": "stdio",
        "tests": [
            {"input": "2 3", "output": "5", "comparison": "nemo_v1"},
            {"input": "4 5", "output": "9", "comparison": "nemo_v1"},
        ],
    }
    assert name == "complete_code" and stratum == "livecodebench_v1:stdio"


def test_invalid_source_assertion_is_a_target_error():
    with pytest.raises(ValueError, match="source test"):
        complete_code.validate_target({"mode": "assert", "tests": ["assert f( == 1"]})


@pytest.mark.parametrize(
    ("prediction", "unit_tests", "expected_score"),
    [
        ("print('yes')", {"inputs": [""], "outputs": ["YES"]}, 1),
        ("print('1.0000005')", {"inputs": [""], "outputs": ["1.0"]}, 1),
        ("print('alice')", {"inputs": [""], "outputs": ["Alice"]}, 0),
        ("print('1.01')", {"inputs": [""], "outputs": ["1.0"]}, 0),
        ("def f(x): return 1.0000005", {"inputs": ["0"], "outputs": ["1.0"], "fn_name": "f"}, 1),
        ("def f(x): return 1.01", {"inputs": ["0"], "outputs": ["1.0"], "fn_name": "f"}, 0),
        ("broken syntax!", {"inputs": [""], "outputs": ["YES"]}, 0),
    ],
)
def test_nemotron_adapter_selects_native_comparison_over_http(monkeypatch, prediction, unit_tests, expected_score):
    service = pytest.importorskip("code_execution_service")
    handler = importlib.import_module("code_execution_service.handler")
    local_server = importlib.import_module("code_execution_service.local_server")
    monkeypatch.setattr(handler, "_TOKEN", "test")
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    server, url = local_server.serve_in_thread()
    session_type = requests.Session

    class LocalSession(session_type):
        def request(self, method, target_url, **kwargs):
            return super().request(method, target_url.replace("https://service.example", url), **kwargs)

    monkeypatch.setattr(requests, "Session", LocalSession)
    try:
        raw = {"question": "Solve", "verifier_metadata": {"unit_tests": unit_tests}}
        _, target, _, _ = sources.adapt(raw, "livecodebench_v1")
        config = dataclasses.replace(CONFIG, expected_source_sha256=service.SOURCE_SHA256)
        result = asyncio.run(complete_code.CompleteCodeVerifier(config).async_call([], prediction, target))
        assert result.score == expected_score
        assert result.diagnostics["status"] == "ok"
        assert result.diagnostics["tests"] == len(unit_tests["inputs"])
    finally:
        server.shutdown()
        server.server_close()


def test_nemotron_target_rejects_unknown_comparison():
    with pytest.raises(ValueError):
        complete_code.validate_target(
            {"mode": "stdio", "tests": [{"input": "", "output": "YES", "comparison": "typo"}]}
        )
