"""Reject incorrect clipping, missing gradients and optimizer scaling defects."""

import copy

import ep_stress_contract
import pytest
import torch


def fixture():
    grad = torch.tensor([3.0, 4.0])
    factor = 0.1 / (5.0 + 1e-6)
    clipped = grad * factor
    return {
        "before": {"x.grad": grad},
        "after": {"x.grad": clipped},
        "state": {"x.main": torch.ones(2), "x.exp_avg": clipped * 0.1, "x.exp_avg_sq": clipped.square() * 0.05},
        "norm": 5.0,
        "clip_grad": 0.1,
        "betas": (0.9, 0.95),
    }


def test_independent_norm_and_adam_moments():
    result = ep_stress_contract.inspect(fixture())
    assert result["independent_grad_norm"] == 5.0
    assert 0 < result["clip_factor"] < 0.02


@pytest.mark.parametrize("fault", ["norm", "clipping", "moment", "variance", "inventory", "nonfinite"])
def test_rejects_contract_faults(fault):
    evidence = copy.deepcopy(fixture())
    if fault == "norm":
        evidence["norm"] = 10.0
    elif fault == "clipping":
        evidence["after"]["x.grad"] *= 2
    elif fault == "moment":
        evidence["state"]["x.exp_avg"] *= 2
    elif fault == "variance":
        evidence["state"]["x.exp_avg_sq"] *= 2
    elif fault == "inventory":
        evidence["before"] = {}
    else:
        evidence["state"]["x.main"][0] = float("nan")
    with pytest.raises((ValueError, AssertionError)):
        ep_stress_contract.inspect(evidence)


def write_campaign(root, fault=None):
    for variant in ("token", "clipped"):
        folder = root / variant
        folder.mkdir()
        for world in (1, 2):
            for rank in range(world):
                evidence = fixture()
                if variant == "token":
                    evidence["clip_grad"] = 1e9
                    evidence["after"]["x.grad"] = evidence["before"]["x.grad"].clone()
                    evidence["state"]["x.exp_avg"] = evidence["after"]["x.grad"] * 0.1
                    evidence["state"]["x.exp_avg_sq"] = evidence["after"]["x.grad"].square() * 0.05
                evidence.update(world=world, rank=rank, token_average=variant == "token")
                if fault == "replica" and rank == 1:
                    evidence["state"]["x.main"] *= 2
                if fault == "identity" and world == 2:
                    evidence["token_average"] = not evidence["token_average"]
                torch.save(evidence, folder / f"stress-ep{world}-rank{rank}.pt")


def test_complete_stress_campaign(tmp_path):
    write_campaign(tmp_path)
    report = ep_stress_contract.compare(tmp_path)
    assert report["passed"]
    assert len(report["arms"]) == 2


@pytest.mark.parametrize("fault", ["replica", "identity", "missing"])
def test_campaign_rejects_rank_or_identity_failures(tmp_path, fault):
    write_campaign(tmp_path, fault)
    if fault == "missing":
        (tmp_path / "clipped/stress-ep2-rank1.pt").unlink()
    with pytest.raises(AssertionError):
        ep_stress_contract.compare(tmp_path)
    assert (tmp_path / "ep-stress.json").is_file()
