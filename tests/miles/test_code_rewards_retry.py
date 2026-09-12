"""The code-service session retries transient gateway failures on its POSTs."""

import asyncio
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


def _score(monkeypatch, status):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session(status))
    args = SimpleNamespace(code_api_url="https://svc/test_program", code_pass_rate_reward_threshold=0.99)
    return asyncio.run(code_rewards.code_score(args, "def f(): pass", ["assert True"], stdio=True))


def test_client_errors_score_the_sample_zero(monkeypatch):
    assert _score(monkeypatch, 413) == 0.0
    assert _score(monkeypatch, 400) == 0.0
    assert _score(monkeypatch, 500) == 0.0  # the harness raised while running this program


def test_service_errors_still_fail_after_retries(monkeypatch):
    with pytest.raises(RuntimeError, match="code verifier request failed"):
        _score(monkeypatch, 503)
    with pytest.raises(RuntimeError, match="code verifier request failed"):
        _score(monkeypatch, 429)


def test_verifier_reports_diagnostics(monkeypatch):
    monkeypatch.setattr(code_rewards, "_get_session", lambda: _Session(413))
    verifier = code_rewards.CodeVerifier(code_rewards.ServiceConfig(api_url="https://svc/test_program", stdio=True))
    result = asyncio.run(verifier.async_call([], "def f(): pass", ["assert True"]))
    assert result.score == 0.0
    assert result.diagnostics["status"] == "rejected" and result.diagnostics["http_status"] == 413
