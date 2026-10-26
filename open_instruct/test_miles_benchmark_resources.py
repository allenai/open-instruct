"""A failed hardware sample must not silently end the rest of a benchmark's telemetry."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles.execution import benchmark_resources


def process(failure=None):
    child = SimpleNamespace(returncode=None, kill=Mock())

    async def communicate():
        if failure is not None:
            raise failure
        child.returncode = 0
        return b"0, GPU-test, GPU, 1, 2, 3, 4, 5\n", b""

    async def wait():
        child.returncode = -9
        return -9

    child.communicate = communicate
    child.wait = AsyncMock(side_effect=wait)
    return child


@pytest.mark.parametrize("failure", [TimeoutError(), FileNotFoundError("nvidia-smi")])
def test_resource_sampling_records_failure_and_recovers(tmp_path, monkeypatch, failure):
    broken, healthy = process(failure), process()
    launch = AsyncMock(side_effect=[failure if isinstance(failure, OSError) else broken, healthy])
    # TimeoutError is an OSError subclass, but this arm must fail inside communicate.
    if isinstance(failure, TimeoutError):
        launch = AsyncMock(side_effect=[broken, healthy])
    monkeypatch.setattr(benchmark_resources.asyncio, "create_subprocess_exec", launch)
    sleeps = 0

    async def sleep(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(benchmark_resources.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(benchmark_resources.observe_resources(tmp_path))
    rows = [json.loads(line) for line in (tmp_path / "benchmark-resources.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["error_type"] == type(failure).__name__
    assert rows[0]["gpu_csv"] == ""
    assert rows[1]["returncode"] == 0 and "GPU-test" in rows[1]["gpu_csv"]
    assert all("consumer_cpu_seconds" in row for row in rows)
    if isinstance(failure, TimeoutError):
        broken.kill.assert_called_once()
        broken.wait.assert_awaited_once()


def test_resource_sampling_cancellation_reaps_child(tmp_path, monkeypatch):
    child = process(asyncio.CancelledError())
    monkeypatch.setattr(benchmark_resources.asyncio, "create_subprocess_exec", AsyncMock(return_value=child))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(benchmark_resources.observe_resources(tmp_path))
    child.kill.assert_called_once()
    child.wait.assert_awaited_once()
    assert (tmp_path / "benchmark-resources.jsonl").read_text() == ""
