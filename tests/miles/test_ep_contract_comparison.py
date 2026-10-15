"""The native comparator must reject shared failures, not only topology drift."""

import json

import ep_contract
import pytest
import torch


def _write_states(root, fault=None):
    for world in (1, 2):
        for checkpointing in (False, True):
            for mode in ("policy", "auxiliary", "combined"):
                state = {}
                for name in ("blocks.0.router.weight", "blocks.0.experts.weight", "blocks.0.dense.weight"):
                    policy = torch.tensor([0.2, -0.3, 0.7])
                    auxiliary = torch.tensor([0.1, 0.2, -0.1])
                    if fault == "zero_auxiliary":
                        auxiliary.zero_()
                    moment = policy if mode == "policy" else auxiliary if mode == "auxiliary" else policy + auxiliary
                    if fault == "superposition" and mode == "combined":
                        moment = moment * 2
                    if fault == "nonfinite" and mode == "combined" and world == 2:
                        moment[0] = float("nan")
                    state[f"{name}.exp_avg"] = moment
                    state[f"{name}.exp_avg_sq"] = moment.square()
                    state[f"{name}.main"] = torch.ones(3)
                torch.save(state, root / f"ep{world}-{mode}-ac{int(checkpointing)}.pt")


def test_native_comparator_accepts_nonzero_additive_moments(tmp_path):
    _write_states(tmp_path)
    ep_contract.compare(tmp_path)
    report = json.loads((tmp_path / "ep-contract.json").read_text())
    assert report["passed"]
    assert len(report["comparisons"]) == 9
    assert len(report["first_moment_signals"]) == 12
    assert len(report["first_moment_superposition"]) == 4
    assert all(
        value["relative_component_l2_error"] < 1e-6
        for comparison in report["first_moment_superposition"]
        for value in comparison["errors"].values()
    )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [("zero_auxiliary", "zero router"), ("superposition", "superposition"), ("nonfinite", "non-finite")],
)
def test_native_comparator_writes_failure_evidence_before_raising(fault, reason, tmp_path):
    # Every topology has the same defect in the first two cases, so simple
    # EP1-vs-EP2 agreement would incorrectly accept both fixtures.
    _write_states(tmp_path, fault)
    with pytest.raises(AssertionError, match=reason):
        ep_contract.compare(tmp_path)
    report = json.loads((tmp_path / "ep-contract.json").read_text())
    assert not report["passed"]
    assert any(reason in failure for failure in report["failures"])
    if fault != "nonfinite":
        assert all(
            value["relative_l2_error"] == 0
            for comparison in report["comparisons"]
            for value in comparison["errors"].values()
        )
