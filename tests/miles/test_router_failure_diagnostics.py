"""Transport failures expose their cause without retrying or leaking prompts."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from miles.router import router
from starlette.requests import ClientDisconnect


@pytest.mark.asyncio
async def test_failure_retains_cause_and_releases_request_count(caplog):
    calls = []
    finished = []
    failure = httpx.ReadError("connection reset")

    async def fail(*a, **kw):
        calls.append((a, kw))
        raise failure

    instance = router.MilesRouter.__new__(router.MilesRouter)
    instance._use_url = lambda: "http://worker:1"
    instance._finish_url = finished.append
    instance.worker_request_counts = {"http://worker:1": 128}
    instance.client = SimpleNamespace(request=fail)
    with pytest.raises(HTTPException) as caught:
        await instance.do_proxy(SimpleNamespace(method="POST"), "generate", b"private prompt", {})
    assert caught.value.status_code == 503
    assert caught.value.__cause__ is failure
    assert len(calls) == 1
    assert finished == ["http://worker:1"]
    assert "worker=http://worker:1 path=generate active=128" in caplog.text
    assert "connection reset" in caplog.text
    assert "private prompt" not in caplog.text


@pytest.mark.parametrize("seconds", [5, 60])
def test_router_shares_explicit_serving_keep_alive(monkeypatch, seconds):
    monkeypatch.setenv("SGLANG_TIMEOUT_KEEP_ALIVE", str(seconds))
    app = object()
    calls = []
    monkeypatch.setattr(router, "configure_logger_raw", lambda *a: None)
    monkeypatch.setattr(router.setproctitle, "setproctitle", lambda *a: None)
    monkeypatch.setattr(router, "MilesRouter", lambda *a, **kw: SimpleNamespace(app=app))
    monkeypatch.setattr(router.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    router.run_router(SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=1234))
    assert calls == [((app,), dict(host="127.0.0.1", port=1234, log_level="info", timeout_keep_alive=seconds))]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [ClientDisconnect, asyncio.CancelledError])
async def test_incoming_disconnect_is_logged_before_forwarding_and_releases_count(caplog, failure_type):
    calls, finished = [], []
    failure = failure_type()

    async def body():
        raise failure

    async def forward(*args, **kwargs):
        calls.append(True)
        raise AssertionError("A partial incoming body must not be forwarded")

    instance = router.MilesRouter.__new__(router.MilesRouter)
    instance._use_url = lambda: "http://worker:1"
    instance._finish_url = finished.append
    instance.client = SimpleNamespace(request=forward)
    request = SimpleNamespace(method="POST", headers={"x-miles-request-id": "request-17"}, body=body)
    with pytest.raises(failure_type) as caught:
        await instance.do_proxy(request, "generate")
    assert caught.value is failure
    assert calls == [] and finished == ["http://worker:1"]
    assert "request='request-17'" in caplog.text
    assert "phase=reading_request_body" in caplog.text
    assert "Forward started" not in caplog.text


@pytest.mark.asyncio
async def test_forwarding_logs_correlate_without_exposing_payload(caplog):
    caplog.set_level("INFO")
    calls, finished = [], []

    async def forward(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(200, content=b'{"answer":"secret response"}')

    instance = router.MilesRouter.__new__(router.MilesRouter)
    instance._use_url = lambda: "http://worker:1"
    instance._finish_url = finished.append
    instance.client = SimpleNamespace(request=forward)
    result = await instance.do_proxy(
        SimpleNamespace(method="POST"), "generate", b"secret prompt", {"x-miles-request-id": "request-18"}
    )
    assert result["status_code"] == 200
    assert calls[0][2]["headers"]["x-miles-request-id"] == "request-18"
    assert finished == ["http://worker:1"]
    assert "Request received: request='request-18'" in caplog.text
    assert "Forward started: request='request-18'" in caplog.text
    assert "Upstream complete: request='request-18'" in caplog.text
    assert "secret prompt" not in caplog.text and "secret response" not in caplog.text
