"""A checkpoint gate must reject changed probabilities, exported weights or optimizer state."""

import copy
import json

import pytest
from scripts.miles import checkpoint_benchmark as gate
from scripts.miles import launch_checkpoint_benchmark as launcher


def reports():
    snapshot = {
        "model": {"weight": {"sha256": "model"}},
        "optimizer": {f"weight.{suffix}": {"sha256": "before"} for suffix in ("main", "exp_avg", "exp_avg_sq")},
        "optimizer_history": {"losses": [], "grad_norms": [{"sha256": "norm"}]},
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
    for report in (saved, resumed):
        for value in report["snapshots"]["step4"]["optimizer"].values():
            value["sha256"] = "after"
    return saved, resumed


@pytest.mark.parametrize(
    "changed", [None, "probabilities", "export", "optimizer", "optimizer_history", "rng", "source", "no_update"]
)
def test_audit_rejects_divergence_and_separates_speed(tmp_path, changed):
    saved, resumed = reports()
    if changed == "probabilities":
        resumed["score_hashes"]["4"] = ["different"]
    elif changed == "export":
        resumed["exports"]["boundary"]["weight"] = "different"
    elif changed in ("optimizer", "optimizer_history", "rng"):
        resumed["snapshots"]["step4"][changed] = {}
    elif changed == "no_update":
        for report in (saved, resumed):
            report["snapshots"]["step4"]["optimizer"] = copy.deepcopy(saved["snapshots"]["step2"]["optimizer"])
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
