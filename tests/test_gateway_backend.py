"""Unit tests for GatewayBackend (LiteRegistry Podman gateway sandbox backend)."""

import base64
from unittest.mock import MagicMock

import pytest

from open_instruct.environments.backends import (
    _DONE_MARKER,
    _PENDING_MARKER,
    GatewayBackend,
    GatewayBackendError,
    create_backend,
)

GATEWAY_URL = "http://gateway.test:8080"


class FakeGateway:
    """In-memory fake of the gateway's /affinity/* HTTP contract.

    Executes a tiny simulation of the in-container filesystem protocol used by
    GatewayBackend: the launcher exec stores the command, the wait exec
    replays the scripted result, uploads decode base64 stdin.
    """

    def __init__(self):
        self.handshakes = 0
        self.requests = []
        self.scripted_results = []  # (exit_code, stdout, stderr) per launched command
        self.pending_rounds = 0  # how many wait execs answer PENDING first
        self.closed = []
        self.container_counter = 0
        self.fail_next_posts: list[tuple[int, str]] = []  # (status_code, body)
        self.markerless_rounds = 0  # how many wait execs return neither marker (e.g. killed, exit 137)
        self.fail_cleanup_posts: list[tuple[int, str]] = []  # (status_code, body) for rm -rf execs only

    def response(self, status_code, payload):
        response = MagicMock()
        response.status_code = status_code
        if isinstance(payload, dict):
            response.json.return_value = payload
            response.text = str(payload)
        else:
            response.text = payload
        return response

    def post(self, url, json=None, timeout=None):
        self.requests.append((url, json))
        if self.fail_next_posts:
            status, body = self.fail_next_posts.pop(0)
            return self.response(status, body)
        if url.endswith("/affinity/handshake"):
            self.handshakes += 1
            self.container_counter += 1
            cid = f"{'c' * 11}{self.container_counter}"
            return self.response(
                200, {"affinity_id": cid, "container_id": cid, "instance_id": "podman-test", "image": json["image"]}
            )
        if url.endswith("/affinity/close"):
            self.closed.append(json["affinity_id"])
            return self.response(200, {"removed": True})
        assert url.endswith("/affinity/podman")
        command = json["command"]
        if "mkdir" in command and "cat >" in command and "LAUNCHED" in command:
            return self.response(
                200, {"stdout": "LAUNCHED\n", "stderr": "", "exit_code": 0, "success": True, "timed_out": False}
            )
        if _DONE_MARKER in command:
            if self.markerless_rounds > 0:
                self.markerless_rounds -= 1
                return self.response(
                    200, {"stdout": "", "stderr": "", "exit_code": 137, "success": False, "timed_out": True}
                )
            if self.pending_rounds > 0:
                self.pending_rounds -= 1
                return self.response(
                    200,
                    {
                        "stdout": f"{_PENDING_MARKER}\n",
                        "stderr": "",
                        "exit_code": 0,
                        "success": True,
                        "timed_out": False,
                    },
                )
            exit_code, stdout, stderr = self.scripted_results.pop(0)
            return self.response(
                200,
                {
                    "stdout": f"{_DONE_MARKER}\nRC={exit_code}\n{stdout}",
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "success": exit_code == 0,
                    "timed_out": False,
                },
            )
        if "rm -rf" in command:
            if self.fail_cleanup_posts:
                status, body = self.fail_cleanup_posts.pop(0)
                return self.response(status, body)
            return self.response(
                200, {"stdout": "", "stderr": "", "exit_code": 0, "success": True, "timed_out": False}
            )
        raise AssertionError(f"FakeGateway got an unexpected command: {command!r}")


@pytest.fixture()
def backend_and_gateway():
    backend = GatewayBackend(image="test-image", timeout=30, gateway_url=GATEWAY_URL, keepalive_interval_s=0)
    fake = FakeGateway()
    backend._session = fake
    backend.start()
    return backend, fake


def test_requires_gateway_url(monkeypatch):
    monkeypatch.delenv("SWERL_GATEWAY_URL", raising=False)
    with pytest.raises(ValueError, match="gateway URL"):
        GatewayBackend(image="test-image")


def test_gateway_url_from_env(monkeypatch):
    monkeypatch.setenv("SWERL_GATEWAY_URL", "http://from-env:1234/")
    backend = GatewayBackend(image="test-image", keepalive_interval_s=0)
    assert backend._gateway_url == "http://from-env:1234"


def test_create_backend_factory(monkeypatch):
    backend = create_backend("gateway", image="img", gateway_url=GATEWAY_URL, mem_limit="4g", keepalive_interval_s=0)
    assert isinstance(backend, GatewayBackend)
    # Non-gateway backends must not receive gateway_url.
    docker_backend = create_backend("docker", image="img", gateway_url=GATEWAY_URL)
    assert not hasattr(docker_backend, "_gateway_url")


def test_run_command_immediate_done(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.scripted_results.append((0, "hello\n", "warn\n"))
    result = backend.run_command("echo hello")
    assert (result.stdout, result.stderr, result.exit_code) == ("hello\n", "warn\n", 0)
    # launch exec carries the command via stdin, not the command field
    launch = next(payload for _, payload in fake.requests if payload and "LAUNCHED" in payload.get("command", ""))
    assert launch["stdin"] == "echo hello"


def test_run_command_polls_through_pending(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.pending_rounds = 2
    fake.scripted_results.append((3, "slow\n", ""))
    result = backend.run_command("slow-command", timeout=600)
    assert result.exit_code == 3
    assert result.stdout == "slow\n"


def test_run_command_timeout_exit_code(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.scripted_results.append((124, "", ""))
    result = backend.run_command("sleep 999", timeout=5)
    assert result.exit_code == 124
    assert "timed out" in result.stderr


def test_run_command_rehandshakes_on_lost_session(backend_and_gateway):
    backend, fake = backend_and_gateway
    first_container = backend._affinity_id
    fake.fail_next_posts.append((404, "binding expired"))
    fake.scripted_results.append((0, "recovered\n", ""))
    result = backend.run_command("echo recovered")
    assert result.stdout == "recovered\n"
    assert fake.handshakes == 2
    assert backend._affinity_id != first_container


def test_run_command_retries_transient_503(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.fail_next_posts.append((503, "replica busy"))
    fake.scripted_results.append((0, "ok\n", ""))
    result = backend.run_command("echo ok")
    assert result.exit_code == 0


def test_run_command_retries_single_owner_unregistered_503(backend_and_gateway):
    # One roster miss is absorbed by the normal retry: the binding stays bound
    # to the same container, no re-handshake.
    backend, fake = backend_and_gateway
    first_container = backend._affinity_id
    fake.fail_next_posts.append((503, "strict affinity server is no longer registered"))
    fake.scripted_results.append((0, "ok\n", ""))
    result = backend.run_command("echo ok")
    assert result.exit_code == 0
    assert fake.handshakes == 1
    assert backend._affinity_id == first_container


def test_run_command_rehandshakes_on_repeated_owner_unregistered_503(backend_and_gateway):
    # A replica that died/rescheduled stays off the roster until the binding
    # TTL expires (~15 min); a repeated "no longer registered" 503 must be
    # treated like a lost session (re-handshake + retry once), not retried
    # to exhaustion.
    backend, fake = backend_and_gateway
    first_container = backend._affinity_id
    fake.fail_next_posts.extend([(503, "strict affinity server is no longer registered")] * 2)
    fake.scripted_results.append((0, "recovered\n", ""))
    result = backend.run_command("echo recovered")
    assert result.stdout == "recovered\n"
    assert fake.handshakes == 2
    assert backend._affinity_id != first_container
    # The dead binding saw exactly the launcher exec and its single quick retry.
    dead_binding_posts = [payload for _, payload in fake.requests if payload.get("affinity_id") == first_container]
    assert len(dead_binding_posts) == 2


def test_close_tolerates_owner_unregistered_503(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.fail_next_posts.extend([(503, "strict affinity server is no longer registered")] * 2)
    backend.close()  # must not raise; container is gone with its replica
    assert backend._affinity_id is None
    assert fake.closed == []


def test_run_command_cleanup_failure_does_not_fail_command(backend_and_gateway, monkeypatch):
    # The post-completion `rm -rf` is best-effort: a replica that 408s it under
    # load (podman exec latency) must not turn a finished command into an error.
    backend, fake = backend_and_gateway
    monkeypatch.setattr(GatewayBackend, "_RETRY_BASE_DELAY_S", 0.0)
    fake.fail_cleanup_posts.extend([(408, '{"detail":"command timed out"}')] * 4)
    fake.scripted_results.append((0, "done\n", ""))
    result = backend.run_command("echo done")
    assert (result.stdout, result.exit_code) == ("done\n", 0)
    assert not fake.fail_cleanup_posts


def test_run_command_retries_markerless_wait(backend_and_gateway):
    # A wait exec cut short (exit 137, no marker) is re-issued; the command's
    # state lives in files so the retry picks up the real result.
    backend, fake = backend_and_gateway
    fake.markerless_rounds = 2
    fake.scripted_results.append((0, "survived\n", ""))
    result = backend.run_command("echo survived")
    assert result.stdout == "survived\n"


def test_run_command_persistent_markerless_wait_raises(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.markerless_rounds = 10
    with pytest.raises(GatewayBackendError, match="no marker"):
        backend.run_command("echo doomed")


def test_non_retryable_http_error_raises(backend_and_gateway):
    backend, fake = backend_and_gateway
    fake.fail_next_posts.append((422, "validation error"))
    with pytest.raises(GatewayBackendError, match="422"):
        backend.run_command("echo nope")


def test_write_file_small_single_exec(backend_and_gateway):
    backend, fake = backend_and_gateway

    seen = {}
    original_exec = backend._exec

    def spy(command, stdin="", **kwargs):
        if "base64 -d" in command:
            seen["stdin"] = stdin
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        return original_exec(command, stdin=stdin, **kwargs)

    backend._exec = spy
    backend.write_file("/workspace/x.txt", "content")
    assert base64.b64decode(seen["stdin"]) == b"content"


def test_close_releases_binding(backend_and_gateway):
    backend, fake = backend_and_gateway
    container = backend._affinity_id
    backend.close()
    assert fake.closed == [container]
    assert backend._affinity_id is None
    # Second close is a no-op.
    backend.close()
    assert fake.closed == [container]
