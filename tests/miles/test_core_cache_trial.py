"""Bounded launcher, fixed config, exact state and observed cache-reuse gates."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import core_cache_trial as trial
from scripts.miles import launch_core_cache_trial as launch

from open_instruct.miles.config import RunConfig

IMAGE = "01M24E7MSDGN2QFW1T8Z31BCKS"


def fixture(root):
    for name in ("cold", "restored"):
        arm = root / name
        arm.mkdir(parents=True)
        report = {
            "status": "completed",
            "child_returncode": 0,
            "fingerprint": "same-key",
            "private_local_root": f"/tmp/{name}",
            "restore": [{"family": "triton", "status": "hit"}],
        }
        (root / f"{name}-cache.json").write_text(json.dumps(report))
        observed = {
            "status": "completed",
            "completed_optimizer_steps": 1,
            "configuration_matches": True,
            "input_inventory": {"fixture": "same-input"},
            "triton": {"put_calls": 8 if name == "cold" else 0, "group_hits": 0 if name == "cold" else 4},
        }
        (arm / "observation.json").write_text(json.dumps(observed))
        states = {"weight": torch.tensor([1.0, 2.0])}
        payload = {
            "initial_model": states,
            "final_model": {"weight": states["weight"] + 0.1},
            "scores": [
                {"use_replay": replay, "scores": [torch.full((3 + index,), -2.0) for index in range(4)]}
                for replay in (False, True, True)
            ],
        }
        torch.save(payload, arm / "observation.pt")
        torch.save({"before": states, "after": states, "state": states}, arm / "stress-ep1-rank0.pt")
        torch.save([torch.zeros(7 + 2 * i, 2, 2, dtype=torch.int64) for i in range(4)], arm / "routes.pt")


def test_exact_update_and_actual_reuse_gate(tmp_path):
    fixture(tmp_path)
    result = trial.compare(tmp_path)
    assert result["passed"]
    assert all(section["exact"] for section in result["comparisons"].values())
    assert result["triton_reuse"] == {"cold_put_calls": 8, "restored_put_calls": 0, "restored_group_hits": 4}
    assert json.loads((tmp_path / "comparison.json").read_text())["passed"]


@pytest.mark.parametrize(
    "section", ["initial_model", "final_model", "scored_log_probabilities", "before", "after", "state", "routing_ids"]
)
def test_each_numerical_boundary_must_be_exact(tmp_path, section):
    fixture(tmp_path)
    if section in ("initial_model", "final_model", "scored_log_probabilities"):
        path = tmp_path / "restored/observation.pt"
        data = torch.load(path, weights_only=True)
        tensor = data["scores"][0]["scores"][0] if section == "scored_log_probabilities" else data[section]["weight"]
    elif section == "routing_ids":
        path = tmp_path / "restored/routes.pt"
        data = torch.load(path, weights_only=True)
        tensor = data[0]
    else:
        path = tmp_path / "restored/stress-ep1-rank0.pt"
        data = torch.load(path, weights_only=True)
        tensor = data[section]["weight"]
    tensor.flatten()[0] += 1
    torch.save(data, path)
    result = trial.compare(tmp_path)
    assert not result["passed"]
    assert not result["comparisons"][section]["exact"]


@pytest.mark.parametrize(
    "change",
    ["no_hit", "no_read", "same_writes", "key", "private_root", "fixture", "steps", "config", "incomplete_scores"],
)
def test_misleading_cache_or_incomplete_update_evidence_rejected(tmp_path, change):
    fixture(tmp_path)
    path = tmp_path / "restored-cache.json"
    data = json.loads(path.read_text())
    if change == "no_hit":
        data["restore"][0]["status"] = "miss"
    elif change == "key":
        data["fingerprint"] = "different-key"
    elif change == "private_root":
        data["private_local_root"] = "/tmp/cold"
    elif change == "incomplete_scores":
        path = tmp_path / "restored/observation.pt"
        payload = torch.load(path, weights_only=True)
        payload["scores"].pop()
        torch.save(payload, path)
        assert not trial.compare(tmp_path)["passed"]
        return
    else:
        path = tmp_path / "restored/observation.json"
        data = json.loads(path.read_text())
        if change == "no_read":
            data["triton"]["group_hits"] = 0
        elif change == "same_writes":
            data["triton"]["put_calls"] = 8
        elif change == "fixture":
            data["input_inventory"]["fixture"] = "changed"
        elif change == "steps":
            data["completed_optimizer_steps"] = 0
        elif change == "config":
            data["configuration_matches"] = False
    path.write_text(json.dumps(data))
    assert not trial.compare(tmp_path)["passed"]


def test_nonfinite_and_different_inventory_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        trial.tensor_comparison({"x": torch.tensor(float("nan"))}, {"x": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="inventory"):
        trial.tensor_comparison({"x": torch.tensor(1.0)}, {"y": torch.tensor(1.0)})


def test_expected_configuration_roundtrip_is_valid(tmp_path):
    expected = trial.expected_config(tmp_path)
    path = tmp_path / "expected.toml"
    path.write_text(trial.config_document(expected))
    assert RunConfig.load(path) == expected
    assert expected.core.expert_parallel_size == 1
    assert expected.core.router_aux_loss_weight == 0.01
    assert expected.core.router_z_loss_weight == 1e-5
    assert expected.miles["use_rollout_routing_replay"]


def test_launcher_uses_requested_placement_bound_and_ttl():
    spec = launch.specification(IMAGE)
    task = spec["tasks"][0]
    assert task["resources"]["gpuCount"] == 1
    assert task["constraints"] == {"cluster": ["ai2/holmes"]}
    assert task["context"] == {"priority": "urgent", "minRuntime": "30m", "autoResume": False}
    assert task["timeout"] == "45m"
    assert task["datasets"][0]["source"] == {"weka": "oe-training-default"}
    command = task["arguments"][0]
    assert IMAGE in command and "/tmp-30d/" in command
    assert "trap" in command and "/output/" in command
    with pytest.raises(ValueError, match="immutable"):
        launch.specification("user/mutable-image")


def test_orchestrator_runs_two_fresh_commands_with_one_canonical_fixture(tmp_path, monkeypatch):
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append(copy.deepcopy(command))
        if "bootstrap" in command:
            fixture_root = Path(command[-1])
            (fixture_root / "hf").mkdir(parents=True)
            (fixture_root / "hf/config.json").write_text('{"hidden_size":128}')
            (fixture_root / "hf/model.safetensors").write_bytes(b"same fixture bytes")
            (fixture_root / "prompts.jsonl").write_text('{"input":"fixture"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", subprocess_run)
    monkeypatch.setattr(trial, "compare", lambda root: {"passed": True})
    root = tmp_path / "run"
    trial.run(root, image=IMAGE, shared_root=tmp_path / "shared", source_root=tmp_path / "source")
    assert len(calls) == 3
    cold, restored = calls[1:]
    assert cold[cold.index("--mode") + 1] == "cold"
    assert restored[restored.index("--mode") + 1] == "restore"
    assert cold[cold.index("--run-config") + 1] == restored[restored.index("--run-config") + 1]
    assert cold[-1] != restored[-1]
    assert "torchrun" in cold and "--nproc-per-node=1" in cold
    assert "--master-addr=127.0.0.1" in cold
    for name in ("cold", "restored"):
        assert (root / name / "hf/model.safetensors").read_bytes() == b"same fixture bytes"
        assert not (root / name / "routes.pt").exists()
    with pytest.raises(FileExistsError):
        trial.run(root, image=IMAGE, shared_root=tmp_path / "shared", source_root=tmp_path / "source")


def test_failed_child_still_gets_a_failed_comparison_report(tmp_path, monkeypatch):
    def subprocess_run(command, **kwargs):
        if "bootstrap" in command:
            root = Path(command[-1])
            (root / "hf").mkdir(parents=True)
            (root / "hf/config.json").write_text("{}")
            (root / "prompts.jsonl").write_text("{}")
            return
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", subprocess_run)
    root = tmp_path / "run"
    with pytest.raises(subprocess.CalledProcessError):
        trial.run(root, image=IMAGE, shared_root=tmp_path / "shared", source_root=tmp_path / "source")
    assert not json.loads((root / "comparison.json").read_text())["passed"]
