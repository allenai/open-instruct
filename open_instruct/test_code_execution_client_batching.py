"""Client batching and bounded wire decoding, using mocked transport only."""

import base64
import json as json_lib
import random
import string
import zlib
from unittest.mock import Mock

import pytest

from open_instruct.code_utils.execution_client import ClientError, CodeExecutionClient, ServiceError, protocol


def test_time_budget_batches_all_tests_in_order():
    session = Mock()

    def post(url, *, json, **kwargs):
        values = [int(test) for test in json["tests"]]
        return Mock(status_code=200, json=Mock(return_value={"results": values, "runtimes": [0] * len(values)}))

    session.post.side_effect = post
    client = CodeExecutionClient("http://unused", session=session, max_request_seconds=7)
    results, _ = client.test_program("", ["0", "1", "1", "0", "1"], max_execution_time=2)
    assert results == [0, 1, 1, 0, 1]
    assert [len(call.kwargs["json"]["tests"]) for call in session.post.call_args_list] == [2, 2, 1]


def test_oversized_single_test_fails_before_any_request():
    session = Mock()
    noise = "".join(random.Random(0).choices(string.ascii_letters, k=4000))
    client = CodeExecutionClient("http://unused", session=session, max_request_bytes=300, max_batch_tests=1)
    with pytest.raises(ClientError) as error:
        client.test_program("", ["assert True", noise])
    assert error.value.status == 413
    session.post.assert_not_called()


def test_late_batch_failure_never_returns_a_partial_score():
    session = Mock()
    session.post.side_effect = [
        Mock(status_code=200, json=Mock(return_value={"results": [1], "runtimes": [0]})),
        Mock(status_code=503, text="unavailable"),
    ]
    client = CodeExecutionClient("http://unused", session=session, retries=0, max_batch_tests=1)
    with pytest.raises(ServiceError):
        client.test_program("", ["assert True"] * 3)
    assert session.post.call_count == 2


def test_deadline_is_shared_across_batches(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("open_instruct.code_utils.execution_client.time.monotonic", lambda: clock[0])
    session = Mock()

    def post(*args, **kwargs):
        clock[0] += 2
        return Mock(status_code=200, json=Mock(return_value={"results": [1], "runtimes": [0]}))

    session.post.side_effect = post
    client = CodeExecutionClient("http://unused", session=session, max_batch_tests=1, total_timeout=1)
    with pytest.raises(ServiceError, match="deadline"):
        client.test_program("", ["assert True"] * 2)
    assert session.post.call_count == 1


def test_last_response_after_deadline_is_not_accepted(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("open_instruct.code_utils.execution_client.time.monotonic", lambda: clock[0])
    session = Mock()

    def post(*args, **kwargs):
        clock[0] = 2
        return Mock(status_code=200, json=Mock(return_value={"results": [1], "runtimes": [0]}))

    session.post.side_effect = post
    client = CodeExecutionClient("http://unused", session=session, total_timeout=1, retries=0)
    with pytest.raises(ServiceError, match="deadline"):
        client.test_program("", ["assert True"])


def test_encoded_legacy_tests_are_counted_after_decoding():
    session = Mock()
    session.post.return_value = Mock(status_code=200, json=Mock(return_value={"results": [1], "runtimes": [0]}))
    result = CodeExecutionClient("http://unused", session=session).test_program("", '["assert True"]')
    assert result == ([1], [0])


def test_decompressed_size_limit(monkeypatch):
    monkeypatch.setattr(protocol, "MAX_DECODED_TEST_BYTES", 32)
    encoded = base64.b64encode(zlib.compress(json_lib.dumps(["x" * 1000]).encode())).decode()
    with pytest.raises(protocol.BadRequest):
        protocol.decode_tests({"encoding": "zlib+json", "data": encoded})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_nonfinite_time_limits_are_rejected(value):
    with pytest.raises(protocol.BadRequest):
        protocol.clamp_per_test_seconds(value)
