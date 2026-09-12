"""Publication timing split and the profile summary that consumes it."""

import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import launch_publication_profile, publication_profile

from open_instruct.miles import actor, publication


def _record(buffer_bytes, total, details):
    return {
        "buffer_bytes": buffer_bytes,
        "total_seconds": total,
        "buckets": len(details),
        "broadcast_seconds": sum(d["broadcast_seconds"] for d in details),
        "engine_seconds": sum(d["engine_seconds"] for d in details),
        "bucket_details": details,
    }


def _detail(tensors, gigabytes, per_tensor=1e-4, per_gb=0.02, broadcast=0.05):
    return {
        "tensors": tensors,
        "bytes": int(gigabytes * 1e9),
        "broadcast_seconds": broadcast,
        "engine_seconds": per_tensor * tensors + per_gb * gigabytes + 0.01,
    }


def test_summary_attributes_engine_time_to_tensors_versus_bytes():
    small = _record(2**30, 3.0, [_detail(900, 1.0), _detail(40, 1.0), _detail(500, 0.6)])
    large = _record(2**32, 2.0, [_detail(2000, 4.0), _detail(120, 4.0)])
    summary = publication_profile.summarize([small, large, small])
    assert set(summary["by_buffer_bytes"]) == {str(2**30), str(2**32)}
    assert summary["by_buffer_bytes"][str(2**30)]["publications"] == 2
    assert summary["by_buffer_bytes"][str(2**30)]["buckets_per_publication"] == 3
    fit = summary["engine_seconds_regression"]
    assert fit["seconds_per_tensor"] == pytest.approx(1e-4, rel=1e-6)
    assert fit["seconds_per_gigabyte"] == pytest.approx(0.02, rel=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert summary["broadcast_gigabytes_per_second"] == pytest.approx(
        (1.0 + 1.0 + 0.6 + 4.0 + 4.0 + 1.0 + 1.0 + 0.6) / (8 * 0.05)
    )


def test_summary_ignores_records_without_bucket_details():
    summary = publication_profile.summarize([{"buffer_bytes": 1, "total_seconds": 1.0, "buckets": 1}])
    assert summary["by_buffer_bytes"] == {} and summary["engine_seconds_regression"] is None


def test_flattened_updater_records_the_broadcast_and_engine_split(monkeypatch):
    updater = publication.FlattenedDistributedUpdater.__new__(publication.FlattenedDistributedUpdater)
    updater._is_src_rank = True
    updater._group_name = "g"
    updater._model_update_groups = object()
    engine = SimpleNamespace(_make_request=SimpleNamespace(remote=lambda *a, **k: "ref"))
    updater.rollout_engines = [engine]
    clock = iter([10.0, 10.4, 10.9])
    monkeypatch.setattr(publication.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(publication.dist, "broadcast", lambda *a, **k: SimpleNamespace(wait=lambda: None))
    monkeypatch.setattr(publication.ray, "get", lambda refs: [{"success": True} for _ in refs])
    named = [("a", torch.zeros(4, dtype=torch.bfloat16)), ("b", torch.ones(2, 3, dtype=torch.bfloat16))]
    updater.update_bucket_weights(named, weight_version=3)
    assert updater.last_bucket_timing == {
        "tensors": 2,
        "bytes": 8 + 12,
        "broadcast_seconds": pytest.approx(0.4),
        "engine_seconds": pytest.approx(0.5),
    }


def test_flattened_updater_reports_engine_rejection(monkeypatch):
    updater = publication.FlattenedDistributedUpdater.__new__(publication.FlattenedDistributedUpdater)
    updater._is_src_rank = True
    updater._group_name = "g"
    updater._model_update_groups = object()
    updater.rollout_engines = [SimpleNamespace(_make_request=SimpleNamespace(remote=lambda *a, **k: "ref"))]
    monkeypatch.setattr(publication.dist, "broadcast", lambda *a, **k: SimpleNamespace(wait=lambda: None))
    monkeypatch.setattr(publication.ray, "get", lambda refs: [{"success": False, "message": "bad"}])
    with pytest.raises(RuntimeError, match="rejected"):
        updater.update_bucket_weights([("a", torch.zeros(4, dtype=torch.bfloat16))])


def test_configure_publication_validates_and_sets_buffer_size():
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(update_weight_buffer_size=1024**3)
    assert worker.configure_publication(2 * 1024**3) == 2 * 1024**3
    assert worker.args.update_weight_buffer_size == 2 * 1024**3
    for bad in (0, -1, 1.5, True, "1"):
        with pytest.raises(ValueError):
            worker.configure_publication(bad)


def test_specification_records_nccl_transport_and_mounts_weka():
    spec = launch_publication_profile.specification("user/image")
    task = spec["tasks"][0]
    script = task["arguments"][0]
    assert task["resources"]["gpuCount"] == 3
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["datasets"][0]["source"] == {"weka": "oe-training-default"}
    assert "export NCCL_DEBUG=INFO" in script and "NCCL_DEBUG_FILE=/output/nccl/" in script
    assert "test ! -e " in script and "publication_profile.py" in script
    json.dumps(spec)
