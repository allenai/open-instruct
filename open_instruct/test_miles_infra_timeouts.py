"""Infrastructure patience must preserve hard failure and useful warnings."""

import asyncio
import time
from unittest.mock import Mock

import pytest

from open_instruct.miles.infrastructure import infra_timeouts


def test_transport_deadline_scaled_with_sanitized_warning(monkeypatch, caplog):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")

    def delayed(*args, **kwargs):
        time.sleep(0.03)
        return kwargs["timeout"]

    session = Mock()
    session.post.side_effect = delayed
    with caplog.at_level("WARNING"):
        result = infra_timeouts.request(
            session, "post", "http://u:secret@judge:8000/generate?token=private", timeout=0.01, json={"max_tokens": 10}
        )
    assert result == 0.05
    assert session.post.call_args.kwargs["json"] == {"max_tokens": 10}
    assert "judge:8000/generate" in caplog.text
    assert "secret" not in caplog.text
    assert "private" not in caplog.text
    assert "finished" in caplog.text


def test_watch_preserves_operation_exception(monkeypatch):
    monkeypatch.delenv("MILES_INFRA_TIMEOUT_MULTIPLIER", raising=False)
    assert infra_timeouts.seconds(30) == 30
    assert infra_timeouts.seconds(None) is None
    with pytest.raises(RuntimeError, match="original error"), infra_timeouts.watch("failed"):
        raise RuntimeError("original error")


@pytest.mark.parametrize("factor", ["0", "-1", "nan", "inf"])
def test_invalid_multiplier_fails(monkeypatch, factor):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", factor)
    with pytest.raises(ValueError):
        infra_timeouts.seconds(30)


def test_async_wait_warns_and_preserves_cancellation(monkeypatch, caplog):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")

    async def exercise():
        assert (
            await infra_timeouts.wait_for(asyncio.sleep(0.03, result="ready"), 0.01, operation="publication")
            == "ready"
        )
        before = asyncio.all_tasks()
        task = asyncio.create_task(infra_timeouts.wait_for(asyncio.sleep(100), 10, operation="cancelled"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == before
        with pytest.raises(TimeoutError):
            await infra_timeouts.wait_for(asyncio.sleep(100), 0.001, operation="stuck")

    with caplog.at_level("WARNING"):
        asyncio.run(exercise())
    assert "operation=publication" in caplog.text
    assert "deadline_s=0.05" in caplog.text
