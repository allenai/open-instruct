"""Checkpoint export and scheduling survive independent worker failures."""

import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from beaker.exceptions import BeakerExperimentConflict, BeakerWorkloadNotFound
from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName
from safetensors.torch import load_file
from torch.distributed import checkpoint as dcp

from open_instruct.miles.evaluation import checkpoint_eval, evaluation, evaluation_submit


def settings(tmp_path):
    return {
        "root": str(tmp_path),
        "submit_timeout": 1,
        "total_updates": 4,
        "initial": False,
        "final": True,
        "tasks": [{"task": "ifeval", "interval": 1, "generation": {}, "scoring": {}}],
        "export_worker": {
            "image": "test-image",
            "workspace": "ai2/olmo-rl",
            "budget": "ai2/oe-other",
            "cluster": "ai2/saturn",
            "token_secret": "beaker-secret",
        },
        "mounts": [{"weka": "checkpoint-weka", "mount_path": "/weka/checkpoint-weka"}],
    }


def native(tmp_path, update):
    path = tmp_path / "core" / f"rollout_{update - 1:07d}"
    model = path / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_text("metadata")
    (model / "__0_0.distcp").write_text(f"weights at {update}")
    (path / "complete.json").write_text(json.dumps({"clock": {"completed_steps": update}}))
    return path


def test_handoff_survives_retention_and_busy_submitter(tmp_path, monkeypatch):
    first = native(tmp_path, 1)
    native(tmp_path, 2)
    worker_blocked = threading.Event()
    release = threading.Event()
    calls = []

    def submit(path, workspace, name, timeout, *, reconcile):
        calls.append(name)
        task = json.loads(path.read_text())["tasks"][0]
        assert "minRuntime" not in task["context"]
        assert task["context"]["autoResume"]
        assert task["resources"].get("gpuCount", 0) == 0
        assert workspace == "ai2/olmo-rl" and reconcile
        worker_blocked.set()
        assert release.wait(2)
        return {"id": name}

    monkeypatch.setattr(evaluation, "create_experiment", submit)
    coordinator = checkpoint_eval.Coordinator(settings(tmp_path), {"name": "run"})
    coordinator.enqueue(1, "/tokenizer", save=tmp_path, rollout=0)
    assert worker_blocked.wait(2)
    request = coordinator.root / "update-00000001.json"
    saved = Path(json.loads(request.read_text())["native"])
    weight_file = "model/__0_0.distcp"
    assert (saved / weight_file).samefile(first / weight_file)
    (first / weight_file).unlink()
    assert (saved / weight_file).read_text() == "weights at 1"
    coordinator.enqueue(2, "/tokenizer", save=tmp_path, rollout=1)
    release.set()
    coordinator.drain()
    assert len(calls) == 2
    assert len(list(coordinator.root.glob("*.submitted.json"))) == 2


def test_worker_submits_only_after_export_and_recovers_missing_receipt(tmp_path, monkeypatch):
    native(tmp_path, 1)
    coordinator = checkpoint_eval.Coordinator(settings(tmp_path), {"name": "run"})
    coordinator.record(1, "/tokenizer", save=tmp_path, rollout=0)
    path = coordinator.root / "update-00000001.json"
    calls = []

    def export(native_path, tokenizer, output, update):
        assert (native_path / "model/.metadata").is_file()
        assert update == 1
        (output / ".complete").touch()

    def submit(receipt, receipt_path, *, reconcile):
        assert reconcile and (Path(receipt["checkpoint"]) / ".complete").is_file()
        calls.append(receipt["group_id"])
        receipt.update(status="submitted", experiment_id="eval")
        receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(
        checkpoint_eval, "importlib", SimpleNamespace(import_module=lambda name: SimpleNamespace(export=export))
    )
    monkeypatch.setattr(evaluation, "submit", submit)
    checkpoint_eval.run(path)
    checkpoint_eval.run(path)
    assert len(calls) == 1


def test_reconcile_ambiguous_beaker_create():
    client = Mock()
    client.workspace.get.return_value.id = "workspace"
    client.user.get.return_value.name = "author"
    client.workload.get.side_effect = [
        BeakerWorkloadNotFound("author/worker"),
        SimpleNamespace(experiment=SimpleNamespace(workspace_id="workspace", id="existing")),
    ]
    client.experiment.create.side_effect = BeakerExperimentConflict("worker")
    assert evaluation_submit.create(client, "spec", "org/workspace", "worker", reconcile=True) == "existing"
    client.experiment.create.assert_called_once()


def test_moe_export_resumes_and_matches_live_conversion(tmp_path, monkeypatch):
    pytest.importorskip("olmo_core.nn.moe.v2.olmo3")

    exporter = importlib.import_module("open_instruct.miles.evaluation.checkpoint_export")
    torch = exporter.torch
    hf = exporter.Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        attention_hidden_size=32,
        head_dim=8,
        moe_intermediate_size=12,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_rope=False,
        use_head_qk_norm=True,
        layer_types=["full_attention", "full_attention"],
    )

    config = exporter.olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=AttentionBackendName.torch
    )
    model = config.build(init_device="cpu")
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.rand(parameter.shape, generator=generator))
    checkpoint = tmp_path / "native"
    checkpoint.mkdir()
    (checkpoint / "complete.json").write_text(
        json.dumps(
            {"clock": {"completed_steps": 1}, "model_config": config.as_config_dict(), "hf_config": hf.to_dict()}
        )
    )
    dcp.save(
        {f"{key}.main": value.detach().float().reshape(-1) for key, value in model.named_parameters()},
        checkpoint_id=checkpoint / "model",
    )
    output = tmp_path / "hf"
    save_file = exporter.save_file

    def interrupt(tensors, path, **kwargs):
        if "layer-001" in path.name:
            raise OSError("preempted during shard write")
        save_file(tensors, path, **kwargs)

    monkeypatch.setattr(exporter, "save_file", interrupt)
    with pytest.raises(OSError, match="preempted"):
        exporter.export(checkpoint, tmp_path, output, 1)
    assert not (output / ".complete").exists()
    first = output / exporter.shard_name(-1)
    before = first.stat().st_ino
    monkeypatch.setattr(exporter, "save_file", save_file)
    exporter.export(checkpoint, tmp_path, output, 1)
    assert first.stat().st_ino == before
    assert (output / ".complete").is_file()
    actual = {}
    for shard in output.glob("*.safetensors"):
        actual.update(load_file(shard))
    expected = dict(exporter.olmo3.iter_olmo3_moe_hf_state(model, hf))
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
