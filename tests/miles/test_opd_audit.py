"""Reject incorrect OPD signals and preserve frozen weights in native exports."""

import json
from pathlib import Path

import pytest
import torch
from safetensors import torch as safetensors_torch

from open_instruct.miles import eopd_math, opd_audit


def test_advantages_must_match_teacher_signal(tmp_path):
    folder = tmp_path / "debug/train_data"
    folder.mkdir(parents=True)
    data = {
        "log_probs": [torch.tensor([-2.0, -3.0])],
        "teacher_log_probs": [torch.tensor([-1.0, -4.0])],
        "advantages": [torch.tensor([1.0, -1.0])],
    }
    path = folder / "0_0.pt"
    torch.save({"rollout_data": data}, path)
    assert opd_audit.audit_training(tmp_path, 1, 1.0)[0]["max_advantage_error"] == 0
    data["advantages"][0].zero_()
    torch.save({"rollout_data": data}, path)
    with pytest.raises(ValueError, match="teacher signal"):
        opd_audit.audit_training(tmp_path, 1, 1.0)


def test_audit_accepts_rollout_logprobs_as_student(tmp_path):
    """use_rollout_logprobs runs dump rollout_log_probs instead of trainer log_probs."""
    folder = tmp_path / "debug/train_data"
    folder.mkdir(parents=True)
    data = {
        "rollout_log_probs": [torch.tensor([-2.0, -3.0])],
        "teacher_log_probs": [torch.tensor([-1.0, -4.0])],
        "advantages": [torch.tensor([0.5, -0.5])],
    }
    torch.save({"rollout_data": data}, folder / "0_0.pt")
    record = opd_audit.audit_training(tmp_path, 1, 0.5)[0]
    assert record["max_advantage_error"] == 0
    assert record["student_log_probs"] == "rollout_log_probs"


def test_export_preserves_frozen_weights_and_fp32_a_log(tmp_path):
    base, export = tmp_path / "base", tmp_path / "hf-0"
    base.mkdir()
    export.mkdir()
    name = "model.language_model.layers.0.linear_attn.A_log"
    tensors = {name: torch.tensor([0.1]), "model.visual.weight": torch.tensor([42.0])}
    safetensors_torch.save_file(tensors, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in tensors}})
    )
    safetensors_torch.save_file({name: torch.tensor([0.2])}, export / "model.safetensors")
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {name: "model.safetensors"}})
    )
    (export / ".complete").touch()
    result = opd_audit.complete_export(tmp_path, base, 0)
    assert result["changed_tensors"] == result["fp32_a_log_tensors"] == 1
    assert result["frozen_base_tensors"] == ["model.visual.weight"]
    frozen = safetensors_torch.load_file(Path(result["path"]) / "frozen-base.safetensors")
    torch.testing.assert_close(frozen["model.visual.weight"], tensors["model.visual.weight"])


def test_export_of_a_plain_language_model_needs_no_index_or_a_log(tmp_path):
    """Qwen3 bases ship one model.safetensors without an index and without A_log tensors."""
    base, export = tmp_path / "base", tmp_path / "hf-2"
    base.mkdir()
    export.mkdir()
    tensors = {"model.layers.0.mlp.up_proj.weight": torch.tensor([0.1]), "model.norm.weight": torch.tensor([1.0])}
    safetensors_torch.save_file(tensors, base / "model.safetensors")
    exported = {**tensors, "model.layers.0.mlp.up_proj.weight": torch.tensor([0.2])}
    safetensors_torch.save_file(exported, export / "model.safetensors")
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": {key: "model.safetensors" for key in exported}})
    )
    (export / ".complete").touch()
    result = opd_audit.complete_export(tmp_path, base, 2)
    assert result["changed_tensors"] == 1 and result["fp32_a_log_tensors"] == 0
    assert result["frozen_base_tensors"] == [] and result["training_scope"] == "full model"
    assert not (export / "frozen-base.safetensors").exists()
    # A language tensor missing from the export is a failure, not a frozen extra.
    safetensors_torch.save_file({"model.norm.weight": torch.tensor([2.0])}, export / "model.safetensors")
    (export / "model.safetensors.index.json").unlink()
    with pytest.raises(ValueError, match="omitted language model weights"):
        opd_audit.complete_export(tmp_path, base, 2)


def test_export_compares_against_the_newest_earlier_export(tmp_path):
    """Exports are saved every N rollouts, so hf-19 is preceded by hf-9, not hf-18."""
    base = tmp_path / "base"
    base.mkdir()
    name = "model.language_model.layers.0.linear_attn.A_log"
    safetensors_torch.save_file({name: torch.tensor([0.1])}, base / "model.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors"}}))
    for rollout_id, value in ((9, 0.2), (19, 0.2)):
        export = tmp_path / f"hf-{rollout_id}"
        export.mkdir()
        safetensors_torch.save_file({name: torch.tensor([value])}, export / "model.safetensors")
        (export / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 4}, "weight_map": {name: "model.safetensors"}})
        )
        (export / ".complete").touch()
    assert opd_audit.complete_export(tmp_path, base, 9)["previous_export"] is None
    with pytest.raises(ValueError, match="did not change between"):
        opd_audit.complete_export(tmp_path, base, 19)


def test_optimizer_audit_requires_every_nonzero_update(tmp_path):
    path = tmp_path / "training.log"
    path.write_text("step 0: {'train/step': 0, 'train/grad_norm': 0.5, 'train/loss': 1.0}\n")
    assert len(opd_audit.audit_optimizer(tmp_path, 1)) == 1
    with pytest.raises(ValueError, match="Missing optimizer"):
        opd_audit.audit_optimizer(tmp_path, 2)
    path.write_text("step 0: {'train/step': 0, 'train/grad_norm': 0.0}\n")
    with pytest.raises(ValueError, match="zero gradient"):
        opd_audit.audit_optimizer(tmp_path, 1)


def test_eopd_audit_rederives_the_gate_and_requires_the_loss_metrics(tmp_path):
    folder = tmp_path / "debug/train_data"
    folder.mkdir(parents=True)
    log_probs = [[-0.05, -3.0], [-0.7, -0.7]]  # first position peaked, second even -> one gated token
    data = {
        "teacher_log_probs": [torch.tensor([-1.0, -4.0])],
        "metadata": [{"eopd_topk_ids": [[2, 7], [3, 8]], "eopd_topk_logprobs": log_probs}],
    }
    torch.save({"rollout_data": data}, folder / "0_0.pt")
    settings = eopd_math.Settings(top_k=2, alpha=1.0, tau=0.5)
    steps = [{"train/step": 0, "train/eopd_fkl_loss": 0.2, "train/eopd_fkl": 0.3, "train/eopd_gate_frac": 0.5}]
    result = opd_audit.audit_eopd(tmp_path, 1, settings, steps)
    assert result["rollouts"][0]["gate_frac"] == 0.5 and result["rollouts"][0]["tokens"] == 2
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        opd_audit.audit_eopd(tmp_path, 1, settings, [{"train/eopd_gate_frac": 0.5}])
    with pytest.raises(ValueError, match="without a forward-KL"):
        opd_audit.audit_eopd(tmp_path, 1, settings, [dict(steps[0], **{"train/eopd_fkl_loss": 0.0})])
    data["metadata"][0]["eopd_topk_ids"] = [[2, 2], [3, 8]]
    torch.save({"rollout_data": data}, folder / "0_0.pt")
    with pytest.raises(ValueError, match="repeats"):
        opd_audit.audit_eopd(tmp_path, 1, settings, steps)
    del data["metadata"]
    torch.save({"rollout_data": data}, folder / "0_0.pt")
    with pytest.raises(ValueError, match="lacks per-sample"):
        opd_audit.audit_eopd(tmp_path, 1, settings, steps)
