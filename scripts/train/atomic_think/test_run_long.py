import sys

import pytest
from scripts.train.atomic_think import run_long


@pytest.mark.parametrize("arm", ["control", "atomic-safe"])
def test_continuation_keeps_full_horizon_restore_guard_and_bounded_audits(tmp_path, monkeypatch, arm):
    prepared, launched = {}, {}
    monkeypatch.setenv("BEAKER_JOB_ID", "job")
    monkeypatch.setenv("BEAKER_REPLICA_RANK", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_long.py",
            arm,
            str(tmp_path / "fresh"),
            str(tmp_path / "data"),
            str(tmp_path / "cache"),
            str(tmp_path / "old" / arm / "step512"),
            "--manifest-sha256",
            "a" * 64,
        ],
    )

    def prepare(*args, **kwargs):
        prepared.update(kwargs)
        return {"verified": True}

    def train(*args, **kwargs):
        launched.update(kwargs)
        launched["duration"] = args[3]

    monkeypatch.setattr(run_long.run_cached, "prepare", prepare)
    monkeypatch.setattr(run_long.run, "train", train)
    run_long.main()
    assert prepared == {"expected_samples": 188544, "manifest_sha256": "a" * 64}
    assert launched["duration"] == 11768
    assert launched["resume"].name == "step512"
    assert launched["verify_resume"] and launched["offline"]
    assert launched["state_steps"][-1] == 11768
    assert 513 in launched["batch_steps"] and 11768 in launched["batch_steps"]
    assert len(launched["batch_steps"]) < 40
