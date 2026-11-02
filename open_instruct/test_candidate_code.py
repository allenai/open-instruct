"""Complete-test code rewards preserve unknown failures and retry transient 500s."""

import asyncio
from types import SimpleNamespace

import pytest
import requests

from open_instruct.miles.rewards import candidate_code, code_rewards


class Session:
    def __init__(self, statuses=(200,), body=None):
        self.statuses = statuses
        self.body = body
        self.payloads = []

    def post(self, url, *, json, timeout):
        status = self.statuses[min(len(self.payloads), len(self.statuses) - 1)]
        self.payloads.append(json)
        response = requests.Response()
        response.status_code = status
        body = self.body if self.body is not None else {"results": [1] * len(json["tests"])}
        response.json = lambda: body
        return response


def score(monkeypatch, session, tests=None):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: session)
    monkeypatch.setattr(code_rewards, "HTTP_500_RETRY_SLEEPS", (0.0, 0.0))
    verifier = candidate_code.CompleteCodeVerifier(candidate_code.Config(api_url="https://svc/test_program"))
    return asyncio.run(
        verifier.async_call(
            [], "<think>reasoning</think>```python\ndef f(): return 1\n```", tests or ["assert f() == 1"]
        )
    )


def test_transient_500_retries_identical_payload_and_keeps_full_tests(monkeypatch):
    session = Session((500, 200, 500, 500, 200))
    tests = [f"assert f() == {i}" for i in range(33)]
    result = score(monkeypatch, session, tests)
    assert result.score == 1.0
    assert result.diagnostics["http_500_retries"] == 3
    assert result.diagnostics["tests"] == 33
    assert session.payloads[0] == session.payloads[1]
    assert session.payloads[2] == session.payloads[3] == session.payloads[4]
    assert [len(p["tests"]) for p in session.payloads] == [16, 16, 16, 16, 16, 1]
    assert session.payloads[0]["program"] == "def f(): return 1"


def test_persistent_500_raises_after_bounded_retries_even_with_zero_policy(monkeypatch):
    monkeypatch.setenv("OI_MILES_CODE_FAILURE_POLICY", "zero")
    session = Session((500,))
    with pytest.raises(requests.HTTPError):
        score(monkeypatch, session)
    assert len(session.payloads) == 3


def test_client_error_is_not_retried_or_scored(monkeypatch):
    session = Session((413, 200))
    with pytest.raises(requests.HTTPError):
        score(monkeypatch, session)
    assert len(session.payloads) == 1


@pytest.mark.parametrize("body", [{}, [], {"results": []}, {"results": [0.5]}, {"results": ["1"]}])
def test_invalid_service_result_is_not_a_reward(monkeypatch, body):
    with pytest.raises(RuntimeError):
        score(monkeypatch, Session(body=body))


def test_valid_failed_test_is_zero(monkeypatch):
    assert score(monkeypatch, Session(body={"results": [0]})).score == 0.0


def test_all_256_tests_are_executed_including_a_failure_in_the_last_batch(monkeypatch):
    session = Session()
    original = session.post

    def post(*args, **kwargs):
        response = original(*args, **kwargs)
        if len(session.payloads) == 16:
            response.json = lambda: {"results": [1] * 15 + [0]}
        return response

    session.post = post
    result = score(monkeypatch, session, [f"assert f() == {i}" for i in range(256)])
    assert result.score == 0.0 and result.diagnostics["passed"] == 255
    assert len(session.payloads) == 16


def test_transport_error_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ReadTimeout("service unavailable")

    with pytest.raises(requests.ReadTimeout):
        score(monkeypatch, SimpleNamespace(post=fail))


@pytest.mark.parametrize("tests", [[], [""], ["assert True"] * 257, "assert True"])
def test_invalid_targets_fail_before_request(tests):
    with pytest.raises(ValueError):
        candidate_code.validate_target(tests)
