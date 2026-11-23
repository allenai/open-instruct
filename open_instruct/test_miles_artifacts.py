"""Concurrent and failed JSON publication must preserve complete receipts."""

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from open_instruct.miles.evaluation import evaluation_runner
from open_instruct.miles.execution import workflow
from open_instruct.miles.infrastructure import artifacts as state

WRITERS = [state.atomic_json, workflow.write_json, evaluation_runner.write_json]


@pytest.mark.parametrize("write", WRITERS)
def test_atomic_json_creates_parent_and_publishes(tmp_path, write):
    target = tmp_path / "run" / "receipt.json"
    write(target, {"step": 1})
    assert json.loads(target.read_text()) == {"step": 1}
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("write", WRITERS)
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


@pytest.mark.parametrize("write", WRITERS)
@pytest.mark.parametrize("failure", ["serialization", "fsync", "replace"])
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


@pytest.mark.parametrize("write", WRITERS)
@pytest.mark.parametrize("directory_failure", [False, True])
def test_atomic_json_syncs_contents_before_publication_and_directory_after(
    tmp_path, monkeypatch, write, directory_failure
):
    target = tmp_path / "receipt.json"
    target.write_text('{"step": 0}')
    synced = []
    descriptors = []
    fsync = os.fsync

    def sync(descriptor):
        descriptors.append(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert synced == ["file"]
            assert json.loads(target.read_text()) == {"step": 1}
            synced.append("directory")
            if directory_failure:
                raise OSError("injected directory sync failure")
        else:
            assert json.loads(target.read_text()) == {"step": 0}
            (temporary,) = tmp_path.glob("*.tmp")
            assert json.loads(temporary.read_text()) == {"step": 1}
            synced.append("file")
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    if directory_failure:
        # Rename has already published the new receipt; failure cannot roll it back.
        with pytest.raises(OSError, match="directory sync failure"):
            write(target, {"step": 1})
    else:
        write(target, {"step": 1})
    assert synced == ["file", "directory"]
    assert json.loads(target.read_text()) == {"step": 1}
    assert list(tmp_path.iterdir()) == [target]
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
