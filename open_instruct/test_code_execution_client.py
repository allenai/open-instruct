"""Client retry behavior without real HTTP calls or backoff waits."""

import json
import threading
from unittest.mock import Mock

import pytest

from open_instruct.code_utils.execution_client import ClientError, CodeExecutionClient, ServiceError


def response(status, payload=None):
    return Mock(status_code=status, text="response", json=Mock(return_value=payload))


def test_throttling_retries_and_can_recover():
    session = Mock()
    session.post.side_effect = [response(429), response(200, {"results": [1], "runtimes": [0.01]})]
    client = CodeExecutionClient("http://unused", session=session, retries=2, backoff=0)
    assert client.test_program("", ["assert True"]) == ([1], [0.01])
    assert session.post.call_count == 2
    request_ids = [call.kwargs["headers"]["X-Request-Id"] for call in session.post.call_args_list]
    assert len(set(request_ids)) == 1


def test_exhausted_throttling_is_a_service_error():
    session = Mock()
    session.post.return_value = response(429)
    client = CodeExecutionClient("http://unused", session=session, retries=2, backoff=0)
    with pytest.raises(ServiceError) as caught:
        client.execute("assert", "", ["assert True"])
    assert session.post.call_count == 3
    assert caught.value.status == 429
    assert caught.value.attempts == 3
    assert caught.value.request_id


def test_invalid_authentication_still_fails_without_retry():
    session = Mock()
    session.post.return_value = response(401)
    client = CodeExecutionClient("http://unused", session=session, retries=2, backoff=0)
    with pytest.raises(ClientError):
        client.execute("assert", "", ["assert True"])
    assert session.post.call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"results": [True], "runtimes": [0]},
        {"results": [1], "runtimes": [float("nan")]},
        {"results": [1], "runtimes": []},
    ],
)
def test_invalid_response_retries_then_recovers(payload):
    session = Mock()
    session.post.side_effect = [response(200, payload), response(200, {"results": [1], "runtimes": [0.01]})]
    client = CodeExecutionClient("http://unused", session=session, retries=1, backoff=0)
    assert client.test_program("", ["assert True"]) == ([1], [0.01])


def test_invalid_json_is_service_failure():
    session = Mock()
    session.post.return_value = response(200)
    session.post.return_value.json.side_effect = ValueError("invalid JSON")
    with pytest.raises(ServiceError, match="invalid grading response"):
        CodeExecutionClient("http://unused", session=session, retries=0).test_program("", ["assert True"])


def test_wrong_deployment_fingerprint_is_not_a_grade():
    session = Mock()
    session.post.return_value = response(200, {"results": [1], "runtimes": [0]})
    session.post.return_value.headers = {"X-Service-Source-Sha256": "wrong"}
    client = CodeExecutionClient("http://unused", session=session, retries=0, expected_source_sha256="expected")
    with pytest.raises(ServiceError, match="fingerprint"):
        client.test_program("", ["assert True"])


def test_retry_jitter_varies_between_callers_and_preserves_recovery(monkeypatch):
    delays = iter([0.1, 0.9])
    waits = []

    def jitter(low, high):
        assert low == 0 and 0.99 <= high <= 1
        return next(delays)

    monkeypatch.setattr("open_instruct.code_utils.execution_client.random.uniform", jitter)
    monkeypatch.setattr("open_instruct.code_utils.execution_client.time.sleep", waits.append)
    for _ in range(2):
        session = Mock()
        session.post.side_effect = [response(503), response(200, {"results": [1], "runtimes": [0.01]})]
        client = CodeExecutionClient("http://unused", session=session, retries=1)
        assert client.test_program("", ["assert True"]) == ([1], [0.01])
    assert waits == [0.1, 0.9]


def test_retry_wait_cannot_extend_the_complete_suite_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("open_instruct.code_utils.execution_client.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("open_instruct.code_utils.execution_client.random.uniform", lambda low, high: high)
    monkeypatch.setattr(
        "open_instruct.code_utils.execution_client.time.sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    session = Mock()
    session.post.return_value = response(503)
    client = CodeExecutionClient("http://unused", session=session, retries=5, backoff=10, total_timeout=0.25)
    with pytest.raises(ServiceError, match="deadline"):
        client.test_program("", ["assert True"])
    assert clock[0] == 0.25
    assert session.post.call_count == 1


def test_cancelled_client_does_not_start_an_http_request():
    cancelled = threading.Event()
    cancelled.set()
    session = Mock()
    with pytest.raises(ServiceError, match="cancelled"):
        CodeExecutionClient("http://unused", session=session, cancel_event=cancelled).test_program("", ["assert True"])
    session.post.assert_not_called()


def test_cancellation_interrupts_retry_wait(monkeypatch):
    entered_wait, finished = threading.Event(), threading.Event()
    errors = []

    class SignalEvent(threading.Event):
        def wait(self, timeout=None):
            entered_wait.set()
            return super().wait(timeout)

    cancelled = SignalEvent()
    session = Mock()
    session.post.return_value = response(503)
    monkeypatch.setattr("open_instruct.code_utils.execution_client.random.uniform", lambda low, high: high)
    client = CodeExecutionClient("http://unused", session=session, cancel_event=cancelled, backoff=60)

    def run():
        try:
            client.test_program("", ["assert True"])
        except ServiceError as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered_wait.wait(3)
        cancelled.set()
        assert finished.wait(3)
        assert len(errors) == 1 and "cancelled" in str(errors[0])
        assert session.post.call_count == 1
    finally:
        cancelled.set()
        worker.join(3)


@pytest.mark.parametrize("status", [401, 503])
def test_bearer_token_is_header_only_and_redacted_from_errors(status):
    token = "synthetic-bearer-for-test-only"
    session = Mock()
    session.post.return_value = response(status)
    session.post.return_value.text = "authorization rejected: Bearer " + token
    client = CodeExecutionClient("https://service.example", token=token, session=session, retries=0)
    with pytest.raises((ClientError, ServiceError)) as caught:
        client.execute("assert", "x=1", ["assert x == 1"])
    request = session.post.call_args.kwargs
    assert request["headers"]["Authorization"] == "Bearer " + token
    assert token not in json.dumps(request["json"])
    assert token not in str(caught.value)
    assert token not in repr(caught.value)
    assert "<redacted>" in str(caught.value)
