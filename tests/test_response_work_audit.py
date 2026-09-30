"""Durable per-rank evidence survives identical Ray log suppression."""

import ast
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "open_instruct"


@pytest.fixture
def audit(monkeypatch):
    spec = importlib.util.spec_from_file_location("response_work_audit", SOURCE / "response_work_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("OI_PACKING_AUDIT", "1")
    monkeypatch.setenv("BEAKER_JOB_ID", "test-job")
    monkeypatch.delenv("OI_PACKING_AUDIT_RESULT_DIR", raising=False)
    return module


def rows(directory):
    return [json.loads(line) for path in directory.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_all_rank_steps_persist_with_no_console_dependency(audit, tmp_path, monkeypatch):
    output, result = tmp_path / "run", tmp_path / "result"
    monkeypatch.setenv("OI_PACKING_AUDIT_RESULT_DIR", str(result))
    records = [
        {"training_step": step, "selected_work": {"tokens": 10, "sample_ids": [rank]}}
        for rank in range(3)
        for step in (1, 2)
    ]
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [
            workers.submit(audit.record, output, "trainer-response-work", payload, rank=index // 2)
            for index, payload in enumerate(records)
        ]
        for future in futures:
            assert future.result()["beaker_job_id"] == "test-job"
    primary = rows(output / "response-work-audit")
    assert len(primary) == 6
    assert {(row["rank"], row["training_step"]) for row in primary} == {
        (rank, step) for rank in range(3) for step in (1, 2)
    }
    assert sorted(primary, key=lambda row: (row["rank"], row["training_step"])) == sorted(
        rows(result), key=lambda row: (row["rank"], row["training_step"])
    )
    # Rank 1 is retained even if its console message is identical to other ranks.
    assert sum(row["rank"] == 1 for row in primary) == 2


def test_preparation_and_selection_are_distinct_from_acceptance(audit, tmp_path):
    data = {
        "data_step": 0,
        "received": {"sample_ids": [0, 1], "tokens": 8},
        "prepared": {"sample_ids": [0], "tokens": 3},
        "dropped_sample_ids": [1],
    }
    audit.record(tmp_path, "packing-retention", data)
    engine = SimpleNamespace(global_steps=4, skipped_steps=1, optimizer=SimpleNamespace(overflow=True))
    counters = audit.engine_counters(engine)
    audit.record(tmp_path, "optimizer-call", {"training_step": 1, "before": counters, "after": counters}, rank=1)
    persisted = rows(tmp_path / "response-work-audit")
    assert len(persisted) == 2 and all("accepted" not in row for row in persisted)
    assert next(row for row in persisted if row["event"] == "packing-retention")["dropped_sample_ids"] == [1]
    assert counters == {"global_steps": 4, "skipped_steps": 1, "optimizer_overflow": True, "native_step_applied": None}
    assert engine.global_steps == 4 and engine.skipped_steps == 1
    assert audit.engine_counters(SimpleNamespace()) == {
        "native_step_applied": None,
        "global_steps": None,
        "skipped_steps": None,
        "optimizer_overflow": None,
    }


def test_flush_failure_fails_closed(audit, tmp_path, monkeypatch):
    def fail(_):
        raise OSError("durability failure")

    monkeypatch.setattr(audit.os, "fsync", fail)
    with pytest.raises(OSError, match="durability failure"):
        audit.record(tmp_path, "trainer-response-work", {"training_step": 1}, rank=1)


def test_disabled_has_no_filesystem_effect(audit, tmp_path, monkeypatch):
    monkeypatch.setenv("OI_PACKING_AUDIT", "0")
    assert audit.record(tmp_path / "missing", "invalid event", {}) is None
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    "event,payload,rank",
    [
        ("../escape", {}, None),
        ("valid", {"event": "replace"}, None),
        ("valid", {"rank": 2}, 1),
        ("valid", {}, -1),
        ("valid", {"tokens": float("nan")}, None),
    ],
)
def test_bad_records_are_rejected_before_directory_creation(audit, tmp_path, event, payload, rank):
    output = tmp_path / "missing"
    with pytest.raises(ValueError):
        audit.record(output, event, payload, rank=rank)
    assert not output.exists()


def test_reject_relative_mirror(audit, tmp_path, monkeypatch):
    monkeypatch.setenv("OI_PACKING_AUDIT_RESULT_DIR", "relative-result")
    with pytest.raises(ValueError, match="absolute"):
        audit.record(tmp_path / "run", "packing-retention", {})
    assert not (tmp_path / "run").exists()


def test_native_training_and_preparation_keep_durable_hooks():
    tree = ast.parse((SOURCE / "grpo_fast.py").read_text())
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PolicyTrainerRayProcess"
    )
    step = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    native_steps = [
        node for node in ast.walk(step) if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.model.step"
    ]
    events = [
        node.args[1].value
        for node in ast.walk(step)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "response_work_audit.record"
    ]
    assert len(native_steps) == 2  # Tiled DAPO and ordinary loss branches retain their native calls.
    assert events.count("optimizer-call") == 2 and events.count("trainer-response-work") == 1
    data = ast.parse((SOURCE / "data_loader.py").read_text())
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "response_work_audit.record"
        and node.args[1].value == "packing-retention"
        for node in ast.walk(data)
    )


@pytest.mark.parametrize("applied,overflow,skipped", [(True, False, 0), (False, True, 1), (False, False, 0)])
def test_native_last_step_query_distinguishes_boundary_overflow_and_noop(audit, applied, overflow, skipped):
    calls = []

    def query():
        calls.append("query")
        return applied

    engine = SimpleNamespace(
        global_steps=5, skipped_steps=skipped, optimizer=SimpleNamespace(overflow=overflow), was_step_applied=query
    )
    counters = audit.engine_counters(engine)
    assert counters["native_step_applied"] is applied
    assert counters["optimizer_overflow"] is overflow
    assert counters["global_steps"] == 5  # A global-step value alone never establishes acceptance.
    assert calls == ["query"] and engine.global_steps == 5


@pytest.mark.parametrize("invalid", [1, "true", []])
def test_native_applied_flag_rejects_unexpected_types(audit, invalid):
    engine = SimpleNamespace(was_step_applied=lambda: invalid)
    with pytest.raises(TypeError, match="must return bool"):
        audit.engine_counters(engine)
