"""A checkpoint gate must reject changed probabilities, exported weights or optimizer state."""

import copy
import json

import pytest
from scripts.miles import checkpoint_benchmark as gate
from scripts.miles import launch_checkpoint_benchmark as launcher


def reports():
    snapshot = {
        "model": {"weight": {"sha256": "model"}},
        "optimizer": {"weight.main": {"sha256": "master"}},
        "scheduler": {"step": 2},
        "clock": {"step": 2},
        "trainer_global_step": 2,
        "cursor": {"sample_offset": 2},
        "rng": {"torch": "state"},
    }
    shared = {
        "runtime_lock": {},
        "harness_sha256": "source",
        "scheduler_horizon": 4,
        "world": 1,
        "rank": 0,
        "score_hashes": {"3": ["logprobs3"], "4": ["logprobs4"]},
        "exports": {"boundary": {"weight": "content"}},
    }
    saved = {
        **copy.deepcopy(shared),
        "pid": 1,
        "save_timings": [{"total_seconds": 400}],
        "snapshots": {key: copy.deepcopy(snapshot) for key in ("step2", "after_save2", "step4")},
    }
    resumed = {
        **copy.deepcopy(shared),
        "pid": 2,
        "snapshots": {key: copy.deepcopy(snapshot) for key in ("restored2", "step4")},
    }
    return saved, resumed


@pytest.mark.parametrize("changed", [None, "probabilities", "export", "optimizer", "rng", "source"])
def test_audit_rejects_divergence_and_separates_speed(tmp_path, changed):
    saved, resumed = reports()
    if changed == "probabilities":
        resumed["score_hashes"]["4"] = ["different"]
    elif changed == "export":
        resumed["exports"]["boundary"]["weight"] = "different"
    elif changed in ("optimizer", "rng"):
        resumed["snapshots"]["step4"][changed] = {}
    elif changed == "source":
        resumed["harness_sha256"] = "different"
    for phase, report in (("split", saved), ("resumed", resumed)):
        (tmp_path / phase).mkdir()
        (tmp_path / phase / "rank0.json").write_text(json.dumps(report))
    if changed:
        with pytest.raises(ValueError, match="correctness gate failed"):
            gate.audit(tmp_path, 1)
    else:
        report = gate.audit(tmp_path, 1)
        assert report["correctness_passed"] and not report["performance_passed"]


def test_launch_bounds_and_modes():
    task = launcher.specification("image")["tasks"][0]
    assert task["resources"]["gpuCount"] == 2
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == "60m"
    assert "for mode in baseline balanced" in task["arguments"][0]
    with pytest.raises(ValueError):
        launcher.specification("image", modes=["bogus"])
