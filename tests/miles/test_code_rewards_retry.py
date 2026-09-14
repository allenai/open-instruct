"""The code-service session retries transient gateway failures on its POSTs."""

import asyncio
import json
import threading
from http import server
from types import SimpleNamespace

import pytest
import requests

from open_instruct.miles import code_rewards


def test_retry_policy_covers_the_scoring_posts():
    retry = code_rewards.RETRY
    assert "POST" in retry.allowed_methods
    assert retry.total >= 5
    assert {502, 503, 504} <= set(retry.status_forcelist)
    # Enough backoff to outlast a gateway hiccup, not so much that a dead service hangs a run forever.
    assert 60 <= sum(min(retry.backoff_factor * 2**i, 120) for i in range(retry.total)) <= 600


def test_session_mounts_the_retry_policy():
    session = code_rewards._get_session()
    assert session.get_adapter("https://example.invalid/").max_retries is code_rewards.RETRY


class _Response:
    def __init__(self, status):
        self.status_code = status

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return {"results": [1, 1]}


class _Session:
    def __init__(self, status):
        self.status = status

    def post(self, *args, **kwargs):
        return _Response(self.status)


def _score(monkeypatch, status, failure_policy=None):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session(status))
    args = SimpleNamespace(
        code_api_url="https://svc/test_program",
        code_pass_rate_reward_threshold=0.99,
        code_failure_policy=failure_policy,
    )
    return asyncio.run(code_rewards.code_score(args, "def f(): pass", ["assert True"], stdio=True))


def test_client_errors_score_the_sample_zero(monkeypatch):
    assert _score(monkeypatch, 413) == 0.0
    assert _score(monkeypatch, 400) == 0.0
    assert _score(monkeypatch, 500) == 0.0  # the harness raised while running this program


def test_service_errors_only_fail_in_strict_mode(monkeypatch):
    with pytest.raises(RuntimeError, match="code verifier request failed"):
        _score(monkeypatch, 503, "raise")
    with pytest.raises(RuntimeError, match="code verifier request failed"):
        _score(monkeypatch, 429, "raise")


def test_verifier_reports_diagnostics(monkeypatch):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session(413))
    verifier = code_rewards.CodeVerifier(code_rewards.ServiceConfig(api_url="https://svc/test_program", stdio=True))
    result = asyncio.run(verifier.async_call([], "def f(): pass", ["assert True"]))
    assert result.score == 0.0
    assert result.diagnostics["status"] == "rejected" and result.diagnostics["http_status"] == 413


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_exhausted_service_errors_default_to_zero(monkeypatch, status):
    assert _score(monkeypatch, status) == 0.0


def test_service_error_diagnostics_are_distinct_from_wrong_code(monkeypatch):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session(503))
    verifier = code_rewards.CodeVerifier(code_rewards.ServiceConfig(api_url="https://svc/test_program"))
    result = asyncio.run(verifier.async_call([], "def f(): pass", ["assert True"]))
    assert result.score == 0.0
    assert result.diagnostics["status"] == "service_error"
    assert result.diagnostics["http_status"] == 503
    assert result.diagnostics["error_type"] == "HTTPError"
    assert result.diagnostics["elapsed_seconds"] >= 0


def test_transport_failure_does_not_poison_subsequent_calls(monkeypatch):
    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.ReadTimeout("read timeout=30")
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"results": [1]})

    session = Session()
    monkeypatch.setattr(code_rewards, "_get_session", lambda: session)

    async def run():
        args = SimpleNamespace(code_api_url="https://svc/test_program")
        failed = await code_rewards.execute(args, "pass", ["assert True"])
        healthy = await code_rewards.execute(args, "pass", ["assert True"])
        return failed, healthy

    (zero, diagnostic), (one, healthy) = asyncio.run(run())
    assert zero == 0.0 and diagnostic["error_type"] == "ReadTimeout"
    assert diagnostic["http_status"] is None
    assert one == 1.0 and healthy["status"] == "ok"


@pytest.mark.parametrize("body", [{}, {"results": [float("nan")]}, {"results": ["invalid"]}])
def test_invalid_service_responses_are_counted_as_fallback_zeros(monkeypatch, body):
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: body)
    monkeypatch.setattr(code_rewards, "_get_session", lambda: SimpleNamespace(post=lambda *a, **kw: response))
    score, diagnostic = asyncio.run(code_rewards.execute(SimpleNamespace(), "pass", []))
    assert score == 0.0 and diagnostic["status"] == "service_error"
    assert diagnostic["error_stage"] == "response"


def test_failure_policy_environment_and_validation(monkeypatch):
    monkeypatch.setenv("OI_MILES_CODE_FAILURE_POLICY", "raise")
    assert code_rewards.code_verifier_config(SimpleNamespace()).failure_policy == "raise"
    assert code_rewards.code_verifier_config(SimpleNamespace(code_failure_policy="zero")).failure_policy == "zero"
    monkeypatch.setenv("OI_MILES_CODE_FAILURE_POLICY", "invalid")
    with pytest.raises(ValueError, match="FAILURE_POLICY"):
        asyncio.run(code_rewards.execute(SimpleNamespace(), "pass", []))


def test_cancellation_is_not_converted_to_a_reward(monkeypatch):
    def request(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(code_rewards, "_get_session", lambda: SimpleNamespace(post=request))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(code_rewards.execute(SimpleNamespace(), "pass", []))


@pytest.mark.parametrize("policy", ["zero", "raise"])
def test_real_http_retry_exhaustion_then_success(monkeypatch, policy):
    class Handler(server.BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            type(self).calls += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            status = 503 if type(self).calls <= 2 else 200
            body = json.dumps({"results": [1]}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    # Exercise the actual adapter retry path without production backoff delays.
    session.mount(
        "http://", requests.adapters.HTTPAdapter(max_retries=code_rewards.RETRY.new(total=1, backoff_factor=0))
    )
    monkeypatch.setattr(code_rewards, "_get_session", lambda: session)
    args = SimpleNamespace(
        code_api_url=f"http://127.0.0.1:{httpd.server_port}/test_program", code_failure_policy=policy
    )

    async def run():
        if policy == "raise":
            with pytest.raises(RuntimeError, match="code verifier request failed"):
                await code_rewards.execute(args, "pass", ["assert True"])
        else:
            score, diagnostics = await code_rewards.execute(args, "pass", ["assert True"])
            assert score == 0.0 and diagnostics["status"] == "service_error"
            assert diagnostics["http_status"] == 503
        score, diagnostics = await code_rewards.execute(args, "pass", ["assert True"])
        assert score == 1.0 and diagnostics["status"] == "ok"

    try:
        asyncio.run(run())
        assert Handler.calls == 3
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        session.close()
