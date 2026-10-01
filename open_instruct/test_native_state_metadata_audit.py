"""Real CPU native-shaped checkpoint serialization and bounded failure cases."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_instruct import native_state_metadata_audit as audit


def checkpoint(tmp_path, mutate=None):
    root = tmp_path / "checkpoint"
    folder = root / "global_step3"
    folder.mkdir(parents=True)
    (root / "latest").write_text("global_step3")
    for rank in range(3):
        model = {
            "rank": rank,
            "training_step": 2,
            "dp_world_size": 3,
            "global_steps": 3,
            "skipped_steps": 0,
            "episode": 512,
            "num_total_tokens": 123,
            "module": {"norm": torch.zeros(3)},
            "param_shapes": [{"norm": torch.Size([3])}],
            "dataloader_state": {"training_step": 2, "current_epoch": 0},
            "data_prep_actor_state": {
                "training_step": 2,
                "last_consumed_step": 1,
                "iter_dataloader_state": {"offset": 16},
            },
            "rng_states": {
                "torch_cpu_rng_state": torch.get_rng_state(),
                "numpy_rng_state": (1, 2),
                "python_rng_state": (1, 2),
            },
        }
        zero = {
            "zero_stage": 3,
            "partition_count": 3,
            "overflow": False,
            "optimizer_state_dict": {
                "param_groups": [{"params": [0]}],
                "state": {0: {"step": torch.tensor(2.0), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.zeros(3)}},
            },
            "fp32_flat_groups": [torch.zeros(3)],
        }
        if mutate:
            mutate(rank, model, zero)
        torch.save(model, folder / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt")
        torch.save({"optimizer_state_dict": zero}, folder / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    return root


def test_real_mmap_metadata_readonly_and_no_restore(tmp_path):
    root = checkpoint(tmp_path)
    report = audit.inspect(root, 3, 2)
    assert report["status"] == "native-metadata-readable"
    assert report["inventory_before"] == report["inventory_after"]
    assert len(report["models"]) == len(report["optimizers"]) == 3
    assert [row["global_steps"] for row in report["models"]] == [3, 3, 3]
    assert all(row["adam_step_min"] == 2 for row in report["optimizers"])
    assert not report["restore_verified"]


@pytest.mark.parametrize(
    "fault",
    [
        "rank",
        "client_step",
        "skip",
        "dp",
        "cursor",
        "rng",
        "moments",
        "overflow",
        "partition",
        "zero_stage",
        "counter",
        "shapes",
        "adam_step",
    ],
)
def test_structural_gate_failures(tmp_path, fault):
    def mutate(rank, model, zero):
        if rank != 1:
            return
        if fault == "rank":
            model["rank"] = 0
        elif fault == "client_step":
            model["training_step"] = 1
        elif fault == "skip":
            model["skipped_steps"] = 1
        elif fault == "dp":
            model["dp_world_size"] = 4
        elif fault == "cursor":
            model["data_prep_actor_state"]["last_consumed_step"] = 0
        elif fault == "rng":
            model["rng_states"].pop("python_rng_state")
        elif fault == "moments":
            zero["optimizer_state_dict"]["state"][0].pop("exp_avg")
        elif fault == "overflow":
            zero["overflow"] = True
        elif fault == "partition":
            zero["partition_count"] = 2
        elif fault == "zero_stage":
            zero["zero_stage"] = 2
        elif fault == "counter":
            model["global_steps"] = 4
        elif fault == "shapes":
            model["param_shapes"] = [{"other": torch.Size([3])}]
        elif fault == "adam_step":
            zero["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(float("nan"))

    with pytest.raises(ValueError):
        audit.inspect(checkpoint(tmp_path, mutate), 3, 2)


@pytest.mark.parametrize(
    "fault", ["partial", "extra", "tracker", "symlink", "directory_link", "bound", "old_serialization"]
)
def test_file_inventory_gates(tmp_path, monkeypatch, fault):
    root = checkpoint(tmp_path)
    shard = root / "global_step3/zero_pp_rank_1_mp_rank_00_model_states.pt"
    if fault == "partial":
        shard.unlink()
    elif fault == "extra":
        (root / "global_step3/extra.pt").write_bytes(b"extra")
    elif fault == "tracker":
        (root / "latest").write_text("../other")
    elif fault == "symlink":
        shard.unlink()
        shard.symlink_to(root / "global_step3/zero_pp_rank_0_mp_rank_00_model_states.pt")
    elif fault == "directory_link":
        folder = root / "global_step3"
        other = tmp_path / "other"
        folder.rename(other)
        folder.symlink_to(other, target_is_directory=True)
    elif fault == "bound":
        monkeypatch.setattr(audit, "MAX_FILE_BYTES", 1)
    elif fault == "old_serialization":
        value = torch.load(shard, weights_only=False)
        torch.save(value, shard, _use_new_zipfile_serialization=False)
    with pytest.raises((ValueError, RuntimeError)):
        audit.inspect(root, 3, 2)


def test_changed_file_during_read_rejected(tmp_path, monkeypatch):
    root = checkpoint(tmp_path)
    original = audit.optimizer_metadata
    called = False

    def observe(state, world_size):
        nonlocal called
        if not called:
            called = True
            tracker = root / "latest"
            tracker.write_text("global_step3\n")
        return original(state, world_size)

    monkeypatch.setattr(audit, "optimizer_metadata", observe)
    with pytest.raises(ValueError, match="changed"):
        audit.inspect(root, 3, 2)


def hook_namespace(monkeypatch, status):
    path = Path(__file__).with_name("grpo_fast.py")
    fn = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "audit_native_checkpoint_metadata"
    )
    records = []

    def inspect(*unused):
        if status == "error":
            raise ValueError("Own partial checkpoint")
        return {"status": "native-metadata-readable", "restore_verified": False}

    ns = {
        "os": os,
        "native_state_metadata_audit": SimpleNamespace(inspect=inspect),
        "response_work_audit": SimpleNamespace(record=lambda *args: records.append(args)),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["audit_native_checkpoint_metadata"], records


@pytest.mark.parametrize("status", ["disabled", "success", "error"])
def test_optin_hook_keeps_error_evidence(monkeypatch, status):
    monkeypatch.setenv("OI_NATIVE_STATE_AUDIT", "0" if status == "disabled" else "1")
    monkeypatch.setenv("OI_PACKING_AUDIT", "1")
    hook, records = hook_namespace(monkeypatch, status)
    args = SimpleNamespace(
        checkpoint_state_dir="/own/run/state", output_dir="/own/run/output", world_size=3, num_training_steps=2
    )
    if status == "error":
        with pytest.raises(ValueError):
            hook(args)
    else:
        hook(args)
    assert len(records) == (0 if status == "disabled" else 1)
    if records:
        assert records[0][2]["status"] == ("error" if status == "error" else "native-metadata-readable")
