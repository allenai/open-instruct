"""Warm compiler caches are published after completed training collections while workers stay live."""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles import compiler_cache as cache
from open_instruct.miles import startup_cache

KEY = "a" * 64


@pytest.fixture
def live_worker(tmp_path):
    """A worker-style private cache under /tmp, exactly as setup_worker creates it."""
    local = Path(tempfile.mkdtemp(prefix="core-triton-", dir="/tmp"))
    (local / "triton" / "group").mkdir(parents=True)
    (local / "triton" / "group" / "kernel.so").write_bytes(b"compiled")
    shared = tmp_path / "shared" / "tmp-7d"
    shared.mkdir(parents=True)
    report = dict(
        slot="train-actor-cell0-rank0",
        local=str(local),
        node_id="node-a",
        shared=str(shared),
        fingerprint=KEY,
        restore={"family": "triton", "status": "miss"},
        setup_seconds=0.1,
    )
    try:
        yield local, shared, report
    finally:
        shutil.rmtree(local, ignore_errors=True)


def test_retained_publication_keeps_the_live_cache_and_is_idempotent(live_worker):
    local, shared, report = live_worker
    first = startup_cache.publish_worker(report, retain_local=True)
    assert first["publish"]["status"] == "published"
    assert first["files_after"] == 1
    # The worker is still running: its mutable Triton directory must survive.
    assert (local / "triton" / "group" / "kernel.so").read_bytes() == b"compiled"
    current = (cache.artifact_root(shared, KEY, "triton") / "CURRENT").read_text().strip()
    assert current == first["publish"]["generation"]
    again = startup_cache.publish_worker(report, retain_local=True)
    assert again["publish"]["status"] == "unchanged"
    assert (local / "triton").is_dir()
    # A later kernel publishes a superset generation without touching the old one.
    (local / "triton" / "group" / "kernel2.so").write_bytes(b"compiled-later")
    third = startup_cache.publish_worker(report, retain_local=True)
    assert third["publish"]["status"] == "published" and third["publish"]["files"] == 2
    assert third["publish"]["generation"] != first["publish"]["generation"]
    generations = cache.artifact_root(shared, KEY, "triton") / "generations"
    assert {path.name for path in generations.iterdir()} >= {
        first["publish"]["generation"],
        third["publish"]["generation"],
    }


def test_final_publication_still_removes_the_private_copy(live_worker):
    local, _, report = live_worker
    result = startup_cache.publish_worker(report)
    assert result["publish"]["status"] == "published"
    assert not local.exists()


def test_progress_schedule_first_collection_then_elapsed_time():
    assert startup_cache.progress_due(None, 100)
    assert not startup_cache.progress_due(100, 699)
    assert startup_cache.progress_due(100, 700)
    assert not startup_cache.progress_due(100, 110, interval=30)
    assert startup_cache.progress_due(100, 130, interval=30)


def _policy(tmp_path, workers=1):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for index in range(workers):
        (report_dir / f"worker-{index}.json").write_text(
            json.dumps(dict(slot=f"rank{index}", local=f"/tmp/core-triton-{index}", node_id="n", fingerprint=KEY))
        )
    (report_dir / "cold.json").write_text(json.dumps(dict(slot="cold", restore={"status": "cold"})))
    return dict(report_dir=str(report_dir), shared=str(tmp_path / "shared"))


def test_publish_progress_runs_in_background_without_overlap(tmp_path, monkeypatch):
    calls = []
    release = None

    async def fake_publish_all(workers, *, retain_local=False):
        calls.append((sorted(worker["slot"] for worker in workers), retain_local))
        await release.wait()
        return [{"slot": worker["slot"], "publish": {"status": "published"}} for worker in workers]

    monkeypatch.setattr(startup_cache, "_publish_all", fake_publish_all)
    startup_cache._PROGRESS.clear()
    args = SimpleNamespace(olmo_core_startup_cache=_policy(tmp_path, workers=2), save=str(tmp_path / "save"))

    async def scenario():
        nonlocal release
        release = asyncio.Event()
        first = startup_cache.publish_progress(args, 4)
        assert first is not None and not first.done()
        # Collections that complete while a publication is in flight are counted, not queued.
        assert startup_cache.publish_progress(args, 9) is None
        release.set()
        await first
        assert calls == [(["rank0", "rank1"], True)]
        progress = json.loads((tmp_path / "save" / "compiler-cache-progress.json").read_text())
        assert progress["publications"][0]["rollout_id"] == 4
        slots = {worker["slot"] for worker in progress["publications"][0]["workers"]}
        assert slots == {"rank0", "rank1", "cold"}
        # Arbitrarily many collections do not bypass the elapsed-time limit.
        state = startup_cache._PROGRESS[args.olmo_core_startup_cache["report_dir"]]
        assert state["collections_seen"] == 2 and state["published"] == 1
        for rollout_id in range(10, 100):
            assert startup_cache.publish_progress(args, rollout_id) is None
        release = asyncio.Event()
        release.set()
        state["last_attempt"] -= startup_cache.DEFAULT_PUBLISH_INTERVAL_SECONDS
        task = startup_cache.publish_progress(args, 100)
        assert task is not None
        await task
        assert len(calls) == 2

    asyncio.run(scenario())


def test_finish_cancels_an_in_flight_progress_publication(tmp_path, monkeypatch):
    started = []

    async def slow_publish_all(workers, *, retain_local=False):
        started.append(retain_local)
        if retain_local:
            await asyncio.sleep(3600)
        return []

    monkeypatch.setattr(startup_cache, "_publish_all", slow_publish_all)
    startup_cache._PROGRESS.clear()
    args = SimpleNamespace(olmo_core_startup_cache=_policy(tmp_path), save=str(tmp_path / "save"))

    async def scenario():
        task = startup_cache.publish_progress(args, 0)
        await asyncio.sleep(0)
        await startup_cache.finish(args, success=True)
        assert task.cancelled()
        assert started == [True, False]
        assert args.olmo_core_startup_cache["report_dir"] not in startup_cache._PROGRESS
        report = json.loads((tmp_path / "save" / "compiler-cache.json").read_text())
        assert report["success"] is True

    asyncio.run(scenario())


def test_publish_progress_is_a_no_op_without_cache_policy():
    assert startup_cache.publish_progress(SimpleNamespace(olmo_core_startup_cache=None), 3) is None


def test_unchanged_live_cache_skips_archiving(live_worker, monkeypatch):
    local, _, report = live_worker
    assert startup_cache.publish_worker(report, retain_local=True)["publish"]["status"] == "published"
    original = cache.publish
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "publish", tracked)
    result = startup_cache.publish_worker(report, retain_local=True)
    assert result["publish"]["reason"] == "local_artifacts_unchanged"
    assert not calls
    (local / "triton" / "group" / "new.so").write_bytes(b"new kernel")
    assert startup_cache.publish_worker(report, retain_local=True)["publish"]["status"] == "published"
    assert calls == [True]


def test_storage_limit_stops_worker_publication_until_run_ends(live_worker, monkeypatch):
    local, shared, report = live_worker
    report["max_storage_bytes"] = 0
    result = startup_cache.publish_worker(report, retain_local=True)
    assert result["publish"]["status"] == "storage_limit"
    assert local.is_dir()
    assert not list(shared.rglob("cache.tar.gz"))

    def unexpected(*args, **kwargs):
        pytest.fail("A capped worker attempted another publication")

    monkeypatch.setattr(cache, "publish", unexpected)
    (local / "triton" / "group" / "later.so").write_bytes(b"still compiles locally")
    assert startup_cache.publish_worker(report, retain_local=True)["publish"]["status"] == "storage_limit"
    assert startup_cache.publish_worker(report)["publish"]["status"] == "storage_limit"
    assert not local.exists()


def test_capped_slots_are_not_resubmitted(tmp_path, monkeypatch):
    calls = []

    async def fake_publish_all(workers, *, retain_local=False):
        calls.append([worker["slot"] for worker in workers])
        return [
            {"slot": w["slot"], "publish": {"status": "storage_limit" if w["slot"] == "rank0" else "published"}}
            for w in workers
        ]

    monkeypatch.setattr(startup_cache, "_publish_all", fake_publish_all)
    args = SimpleNamespace(olmo_core_startup_cache=_policy(tmp_path, workers=2), save=str(tmp_path / "save"))

    async def scenario():
        await startup_cache.publish_progress(args, 0)
        state = startup_cache._PROGRESS[args.olmo_core_startup_cache["report_dir"]]
        state["last_attempt"] -= startup_cache.DEFAULT_PUBLISH_INTERVAL_SECONDS
        await startup_cache.publish_progress(args, 1)
        assert calls == [["rank0", "rank1"], ["rank1"]]
        startup_cache._PROGRESS.pop(args.olmo_core_startup_cache["report_dir"])

    asyncio.run(scenario())
