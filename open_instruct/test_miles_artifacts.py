"""Concurrent and failed JSON publication must preserve complete receipts."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from open_instruct.miles.evaluation import evaluation_runner
from open_instruct.miles.infrastructure import artifacts as state


@pytest.mark.parametrize("write", [state.atomic_json, evaluation_runner.write_json])
def test_atomic_json_creates_parent_and_publishes(tmp_path, write):
    target = tmp_path / "run" / "receipt.json"
    write(target, {"step": 1})
    assert json.loads(target.read_text()) == {"step": 1}
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("write", [state.atomic_json, evaluation_runner.write_json])
def test_atomic_json_concurrent_writers(tmp_path, monkeypatch, write):
    target = tmp_path / "receipt.json"
    target.write_text('{"step": 0}')
    barrier = threading.Barrier(2)
    replace = Path.replace
    values = [{"step": 1, "payload": "a" * 4096}, {"step": 2, "payload": "b" * 8192}]

    def publish(source, destination):
        # Both writers must finish their private file before either can publish.
        assert json.loads(target.read_text()) == {"step": 0}
        barrier.wait(timeout=10)
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, target, value) for value in values]
        for future in futures:
            future.result(timeout=15)

    assert json.loads(target.read_text()) in values
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize(
    "write,failure",
    [(state.atomic_json, failure) for failure in ("serialization", "fsync", "replace")]
    + [(evaluation_runner.write_json, failure) for failure in ("serialization", "replace")],
)
def test_atomic_json_failure_preserves_receipt_and_cleans_temp(tmp_path, monkeypatch, write, failure):
    target = tmp_path / "receipt.json"
    original = '{"step": 0}'
    target.write_text(original)
    value = {"step": 1}

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        if failure == "serialization":
            value["invalid"] = object()
        elif failure == "fsync":
            patch.setattr(state.os, "fsync", fail)
        else:
            patch.setattr(Path, "replace", fail)
        with pytest.raises((TypeError, OSError)):
            write(target, value)

    assert target.read_text() == original
    assert list(tmp_path.iterdir()) == [target]
    write(target, {"step": 2})
    assert json.loads(target.read_text()) == {"step": 2}
