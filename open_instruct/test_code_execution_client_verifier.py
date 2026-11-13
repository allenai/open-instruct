"""Extraction and trainer-facing failure policy for the registry adapter."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.code_utils.execution_client import ExecuteResult, Outcome, ServiceError, verifier


@pytest.mark.parametrize("status", [200, 503])
def test_async_cancellation_drains_current_request_without_more_batches_or_retries(monkeypatch, status):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(side_effect=lambda *args: closed.set())

    def post(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return Mock(
            status_code=status,
            text="unavailable",
            json=Mock(
                return_value={
                    "outcomes": [{"status": "passed", "runtime": 0.01, "detail": None}] * 16,
                    "wall_seconds": 0.1,
                    "exit": "ok",
                }
            ),
        )

    session.post.side_effect = post
    monkeypatch.setattr(verifier.requests, "Session", lambda: session)
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local"))

    async def run():
        task = asyncio.create_task(adapter.async_call([], "x=1", ["assert x==1"] * 33))
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closed.is_set()  # The running HTTP call still owns the session.
        finally:
            release.set()
        async with asyncio.timeout(3):
            while not closed.is_set():
                await asyncio.sleep(0.01)

    asyncio.run(run())
    assert session.post.call_count == 1


def test_worker_queue_time_counts_toward_the_complete_suite_deadline(monkeypatch):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    clock = [0.0]
    monkeypatch.setattr(verifier, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def delayed_worker(function):
        clock[0] += 2
        return function()

    monkeypatch.setattr(verifier.asyncio, "to_thread", delayed_worker)
    monkeypatch.setattr(verifier.requests, "Session", lambda: pytest.fail("expired work must not open HTTP session"))
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local", total_timeout=1))
    with pytest.raises(ServiceError, match="waiting for a worker"):
        asyncio.run(adapter.async_call([], "x=1", ["assert x==1"]))


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python\nprint(5)\n```\n```text\n5\n```", "print(5)"),
        ("```python\nprint(1)\n```\n```python3\nprint(5)\n```", "print(5)"),
        ("```\nprint(5)\n```", "print(5)"),
        ("print(5)", "print(5)"),
        ("```javascript\nconsole.log(5)\n```", None),
        ("```python\nprint(1)\n```\n```python\nprint(", None),
        ("~~~~python\nprint(5)\n~~~~", "print(5)"),
        ("````python\nprint('```')\n````", "print('```')"),
        ('```python title="solution.py"\nprint(5)\n```', "print(5)"),
        ("```python\nprint(5)\n```<|endoftext|>", "print(5)"),
        ("```python\nprint(5)\n```<|im_end|>\n<|endoftext|>\n", "print(5)"),
        ("```python\nprint(5)\n```<|eot_id|>", "print(5)"),
        ("print(5)<|endoftext|>", "print(5)"),
        ("```python\nprint(5)\n<|endoftext|>", None),
        ("```python\nprint(5)\n```unexpected", None),
        ('```python\nprint("<|endoftext|>")\n```<|endoftext|>', 'print("<|endoftext|>")'),
        ('value = "<|endoftext|>"', 'value = "<|endoftext|>"'),
        ('value = """\n```text\nhello\n```\n"""\nprint(value)', 'value = """\n```text\nhello\n```\n"""\nprint(value)'),
    ],
)
def test_extracts_python_without_taking_trailing_text(response, expected):
    assert verifier.extract_python_code(response) == expected


def test_registry_adapter_uses_complete_authenticated_execution(monkeypatch):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    captured = {}

    def execute(self, mode, program, tests, **kwargs):
        captured.update(token=self.token, url=self.base_url, program=program, tests=tests)
        return ExecuteResult([Outcome("passed", 0.01, None)], 0.1, "ok", "id", 1, ("id",))

    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", execute)
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local/test_program"))
    result = asyncio.run(adapter.async_call([], "```python\nx=1\n```\n```text\n1\n```", ["assert x==1"]))
    assert result.score == 1
    assert captured == {"token": "test", "url": "http://local", "program": "x=1", "tests": ["assert x==1"]}


def test_registry_adapter_grades_code_followed_by_decoded_end_token(monkeypatch):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    captured = []

    def execute(self, mode, program, tests, **kwargs):
        captured.append(program)
        return ExecuteResult([Outcome("passed", 0.01, None)], 0.1, "ok", "id", 1, ("id",))

    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", execute)
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local"))
    response = '```python\ndef generate_greeting():\n    return "Welcome!"\n```<|endoftext|>'
    result = asyncio.run(adapter.async_call([], response, ['assert generate_greeting() == "Welcome!"']))
    assert result.score == 1.0
    assert result.diagnostics["status"] == "ok"
    assert captured == ['def generate_greeting():\n    return "Welcome!"']


def test_registry_adapter_never_turns_service_failure_into_reward(monkeypatch):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")

    def fail(*args, **kwargs):
        raise ServiceError("unavailable")

    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", fail)
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local"))
    with pytest.raises(ServiceError):
        asyncio.run(adapter.async_call([], "x=1", ["assert x==1"]))


def test_registry_adapter_rejects_empty_test_sets(monkeypatch):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local"))
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(adapter.async_call([], "x=1", []))


@pytest.mark.parametrize("prediction", ["", "   ", "```python\nprint(", "```javascript\nx=1\n```"])
def test_invalid_code_is_a_completed_zero(monkeypatch, prediction):
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", lambda *a, **k: pytest.fail("must not call service"))
    result = asyncio.run(
        verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local")).async_call(
            [], prediction, ["assert True"]
        )
    )
    assert result.score == 0
    assert result.diagnostics == {"status": "completed", "outcome": "invalid_code"}


def test_invalid_code_to_miles_validity_contract(monkeypatch):
    # Optional integration dependency; CI/operators can supply a MILES checkout
    # on PYTHONPATH without adding the training stack to the service package.
    records = pytest.importorskip("miles.utils.inference_records")
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://local"))
    result = asyncio.run(adapter.async_call([], "", ["assert True"]))
    metadata = {"verifiers": [{"name": "code"}], "verifier_diagnostics": {"code": result.diagnostics}}
    assert result.score == 0 and records.validity(metadata)["valid"] is True

    def fail(*args, **kwargs):
        raise ServiceError("unavailable")

    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", fail)
    unknown = {"verifiers": [{"name": "code"}]}
    with pytest.raises(ServiceError):
        asyncio.run(adapter.async_call([], "x=1", ["assert x==1"]))
    assert records.validity(unknown)["valid"] is None
