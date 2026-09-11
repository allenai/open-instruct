"""CPU checks of observation-only hooks against a synthetic contract boundary."""

import json
from types import SimpleNamespace

import cache_core_contract as worker
import pytest
import torch
from scripts.miles import core_cache_trial


def setup_worker(tmp_path, monkeypatch, *, wrong_config=False):
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf/config.json").write_text("{}")
    (tmp_path / "prompts.jsonl").write_text('{"input":"fixture"}\n')
    expected = core_cache_trial.expected_config(tmp_path)
    (tmp_path / "expected.toml").write_text(core_cache_trial.config_document(expected))
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    instance = SimpleNamespace(model=model, clock=SimpleNamespace(completed_steps=0))
    returns = []

    def original_score(actor, module, batches, *, use_replay):
        values = [torch.arange(3 + i).float() for i in range(4)]
        returns.append(values)
        return values

    def production_boundary(root, mode, checkpointing, *, capture_gradients):
        assert mode == "combined" and not checkpointing and capture_gradients
        miles = dict(expected.miles)
        if wrong_config:
            miles["lr"] = 2e-4
        worker.ep_contract.RunConfig(expected.core, miles)
        for replay in (False, True, True):
            result = worker.ep_contract.actor.OLMoCoreTrainRayActor._score(instance, None, None, use_replay=replay)
            assert result is returns[-1], "Observer changed the object consumed by production"
        model.weight.data.add_(0.25)
        instance.clock.completed_steps = 1
        worker.ep_contract.full_optimizer_state(instance)
        torch.save({"placeholder": True}, root / "stress-ep1-rank0.pt")
        destination = root / "ep1-combined-ac0"
        destination.mkdir()
        (destination / "training_contract_rank0.jsonl").write_text(
            json.dumps({"event": "optimizer", "step": 1, "optimizer_skipped": False, "elapsed_seconds": 0.1}) + "\n"
        )

    monkeypatch.setattr(worker.ep_contract.actor.OLMoCoreTrainRayActor, "_score", original_score)
    monkeypatch.setattr(worker.ep_contract, "full_optimizer_state", lambda actor, gradients=False: {})
    monkeypatch.setattr(worker.ep_contract, "run", production_boundary)
    monkeypatch.setattr(worker.ep_stress_contract, "inspect", lambda evidence: {"synthetic_boundary_test_only": True})


def test_observer_preserves_consumed_scores_and_captures_initial_final_states(tmp_path, monkeypatch):
    setup_worker(tmp_path, monkeypatch)
    worker.run(tmp_path)
    report = json.loads((tmp_path / "observation.json").read_text())
    assert report["status"] == "completed" and report["configuration_matches"]
    assert report["completed_optimizer_steps"] == 1
    assert report["triton"]["put_calls"] == 0 and report["triton"]["group_hits"] == 0
    payload = torch.load(tmp_path / "observation.pt", weights_only=True)
    assert [value["use_replay"] for value in payload["scores"]] == [False, True, True]
    assert torch.equal(payload["initial_model"]["weight"], torch.ones(1, 2))
    assert torch.equal(payload["final_model"]["weight"], torch.full((1, 2), 1.25))


def test_actual_constructor_mismatch_fails_before_update(tmp_path, monkeypatch):
    setup_worker(tmp_path, monkeypatch, wrong_config=True)
    with pytest.raises(ValueError, match="fingerprinted"):
        worker.run(tmp_path)
    report = json.loads((tmp_path / "observation.json").read_text())
    assert report["status"] == "failed" and report["completed_optimizer_steps"] == 0
    assert not report["configuration_matches"]
