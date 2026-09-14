"""Transport failures expose their cause without retrying or leaking prompts."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from miles.router import router


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
