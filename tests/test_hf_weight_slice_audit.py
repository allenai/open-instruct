"""Actual CPU safetensor slicing plus isolated launcher integration checks."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as safe_torch


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "open_instruct/hf_weight_slice_audit.py"
    spec = importlib.util.spec_from_file_location("hf_weight_slice_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights():
    return {
        "model.language_model.embed_tokens.weight": torch.arange(960, dtype=torch.float32).reshape(40, 24),
        "model.language_model.norm.weight": torch.ones(40),
        **{
            f"model.language_model.layers.{layer}.{family}.weight": torch.ones(40, 24)
            for layer in (3, 11)
            for family in ("self_attn.q_proj", "mlp.down_proj")
        },
    }


def write_weights(root, tensors=None):
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen3_5"}')
    safe_torch.save_file(weights() if tensors is None else tensors, root / "model.safetensors")
    return root


def test_actual_finite_slice_movement_and_readonly(audit, tmp_path):
    original = write_weights(tmp_path / "initial")
    final_values = weights()
    key = "model.language_model.layers.11.self_attn.q_proj.weight"
    final_values[key][16, 4] += 0.125
    # This unsampled change must not be incorrectly counted.
    final_values[key][9, 23] += 10
    final = write_weights(tmp_path / "final", final_values)
    report = audit.compare(original, final)
    assert report["sampled_weight_movement"] and report["source_stats_unchanged"]
    assert len(report["slices"]) == 6
    changed = [record for record in report["slices"] if record["changed_sampled_elements"]]
    assert len(changed) == 1
    assert changed[0]["changed_sampled_elements"] == 1 and changed[0]["max_abs_gap"] == 0.125
    assert changed[0]["row_starts"] == [0, 16, 32]
    assert sum(record["sampled_elements"] for record in report["slices"]) == 1944


def test_equal_slices_do_not_assert_full_equality(audit, tmp_path):
    original, final = write_weights(tmp_path / "initial"), write_weights(tmp_path / "final")
    report = audit.compare(original, final)
    assert not report["sampled_weight_movement"]
    assert "not full equality" in report["limits"]


def test_text_families_exclude_visual_norm_and_attention(audit):
    mapping = weights() | {"model.visual.norm.weight": None, "model.visual.layers.0.self_attn.q_proj.weight": None}
    selected = audit.selected_keys(mapping)
    assert len(selected) == 6 and all(key.startswith("model.language_model.") for key in selected)


@pytest.mark.parametrize("change", ["nonfinite", "shape", "missing"])
def test_bad_exports_rejected(audit, tmp_path, change):
    original = write_weights(tmp_path / "initial")
    tensors = weights()
    key = "model.language_model.norm.weight"
    if change == "nonfinite":
        tensors[key][0] = float("nan")
    elif change == "shape":
        tensors[key] = torch.ones(41)
    else:
        del tensors[key]
    final = write_weights(tmp_path / "final", tensors)
    with pytest.raises(ValueError):
        audit.compare(original, final)


def test_header_index_path_escape_and_bound(audit, tmp_path):
    root = write_weights(tmp_path / "weights")
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {key: "../model.safetensors" for key in weights()}}))
    with pytest.raises(ValueError, match="plain filename"):
        audit.inventory(root)
    index.write_text(" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds bound"):
        audit.inventory(root)


def test_indexed_shards_and_only_expected_cache_links(audit, tmp_path):
    root = write_weights(tmp_path / "model-cache" / "snapshots" / "revision")
    blobs = root.parent.parent / "blobs"
    blobs.mkdir()
    shard = root / "model.safetensors"
    shard.rename(blobs / "immutable-blob")
    shard.symlink_to(blobs / "immutable-blob")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard.name for key in weights()}})
    )
    with pytest.raises(ValueError, match="symlink"):
        audit.inventory(root)
    assert len(audit.inventory(root, cache_links=True)[0]) == 6
    shard.unlink()
    outside = tmp_path / "outside"
    safe_torch.save_file(weights(), outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        audit.inventory(root, cache_links=True)


def test_input_change_during_sampling_rejected(audit, tmp_path, monkeypatch):
    original, final = write_weights(tmp_path / "initial"), write_weights(tmp_path / "final")
    native = audit.read_slice

    def mutating_read(*args, **kwargs):
        result = native(*args, **kwargs)
        (original / "config.json").write_text('{"changed":true}')
        return result

    monkeypatch.setattr(audit, "read_slice", mutating_read)
    with pytest.raises(ValueError, match="inventory changed"):
        audit.compare(original, final)


def test_overlapping_roots_rejected(audit, tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        audit.compare(tmp_path, tmp_path / "nested")


def test_main_hook_offline_binding_record_and_rejection(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "open_instruct/grpo_fast.py"
    tree = ast.parse(source.read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "audit_final_weight_slices"
    )
    revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    original = tmp_path / "snapshots" / revision
    original.mkdir(parents=True)
    final = tmp_path / "export"
    final.mkdir()
    (final / "complete").touch()
    calls, records = [], []
    report = {"sampled_weight_movement": True}
    namespace = {
        "os": SimpleNamespace(environ={"OI_FINAL_WEIGHT_AUDIT": "1", "OI_PACKING_AUDIT": "1"}),
        "Path": Path,
        "CHECKPOINT_COMPLETE_MARKER": "complete",
        "snapshot_download": lambda *args, **kwargs: calls.append((args, kwargs)) or str(original),
        "hf_weight_slice_audit": SimpleNamespace(compare=lambda *args, **kwargs: report),
        "response_work_audit": SimpleNamespace(record=lambda *args: records.append(args)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    config = SimpleNamespace(model_name_or_path="Qwen/Qwen3.5-2B", model_revision=revision)
    args = SimpleNamespace(output_dir=str(final))
    hook = namespace["audit_final_weight_slices"]
    hook(args, config)
    assert calls[0][1] == {"revision": revision, "local_files_only": True}
    assert records[0] == (str(final), "final-weight-slices", report)
    report["sampled_weight_movement"] = False
    with pytest.raises(RuntimeError, match="not established"):
        hook(args, config)
    config.model_revision = "wrong"
    with pytest.raises(ValueError, match="pinned"):
        hook(args, config)
    namespace["os"].environ.clear()
    hook(None, None)
