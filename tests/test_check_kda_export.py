"""scripts/train/debug/check_kda_export.py passes an export whose KDA weights hold what --kda-decay says, and fails
each kind of wrong one. Tiny checkpoint and export written here; no model is built."""

import importlib.util
import json
import pathlib

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train" / "debug" / "check_kda_export.py"
spec = importlib.util.spec_from_file_location("check_kda_export", SCRIPT)
assert spec is not None and spec.loader is not None
check_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_script)

WEIGHTS = (*check_script.DECAY, *check_script.CONVS)
KDA_LAYERS = (0, 2)  # layer 1 is full attention


def masters() -> dict[tuple[int, str], torch.Tensor]:
    """Float32 values that bfloat16 cannot represent, different for every tensor."""
    generator = torch.Generator().manual_seed(0)
    values = {}
    for layer in KDA_LAYERS:
        for weight in WEIGHTS:
            shape = (4, 1, 4) if weight in check_script.CONVS else (4,)
            value = 1.0 + torch.rand(shape, generator=generator) + 2.0**-12
            assert (value != value.to(torch.bfloat16).to(torch.float32)).any()
            values[layer, weight] = value
    return values


def write_checkpoint(path: pathlib.Path, values, key: str) -> pathlib.Path:
    state = {key.format(name=f"blocks.{layer}.attention.{weight}"): value for (layer, weight), value in values.items()}
    state[key.format(name="blocks.1.attention.w_q.weight")] = torch.ones(2, 2)
    dist_cp.save(state, checkpoint_id=str(path / "model_and_optim"))
    return path


def write_export(path: pathlib.Path, values, kda_decay: str, change=None) -> pathlib.Path:
    """An export as convert_moe_checkpoint_to_hf.py writes it in ``kda_decay`` mode; ``change`` edits one tensor."""
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"layer_types": ["linear_attention", "full_attention", "linear_attention"]})
    )
    tensors = {"model.layers.1.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    for (layer, weight), value in values.items():
        on_grid = value.to(torch.bfloat16)
        if kda_decay == "fp32":
            tensor = value.clone()
        else:
            tensor = on_grid.to(torch.float32) if weight in check_script.DECAY else on_grid
        tensors[f"model.layers.{layer}.linear_attn.{weight}"] = tensor
    if change:
        name, edit = change
        tensors[name] = edit(tensors[name])
    half = len(tensors) // 2  # two shards, as a large export has
    items = list(tensors.items())
    save_file(dict(items[:half]), str(path / "model-00001-of-00002.safetensors"))
    save_file(dict(items[half:]), str(path / "model-00002-of-00002.safetensors"))
    return path


def run(monkeypatch, checkpoint, hf_dir, kda_decay, output) -> tuple[int, dict]:
    argv = ["check", "--checkpoint", str(checkpoint), "--hf-dir", str(hf_dir), "--kda-decay", kda_decay]
    monkeypatch.setattr("sys.argv", [*argv, "--output", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        check_script.main()
    return exit_info.value.code, json.loads(output.read_text())


@pytest.mark.parametrize("key", ["model.{name}", "{name}.main"])
@pytest.mark.parametrize("kda_decay", check_script.MODES)
def test_right_export_passes(tmp_path, monkeypatch, kda_decay, key):
    values = masters()
    checkpoint = write_checkpoint(tmp_path / "step1", values, key)
    hf_dir = write_export(tmp_path / "hf", values, kda_decay)
    code, result = run(monkeypatch, checkpoint, hf_dir, kda_decay, tmp_path / "result.json")
    assert code == 0 and result["ok"] and result["failures"] == []
    assert (result["kda_layers"], result["tensors"]) == (2, 10)


def test_fp32_fails_an_export_with_rounded_weights(tmp_path, monkeypatch):
    """The defect this check exists for: an export said to hold float32 masters whose values were rounded."""
    values = masters()
    checkpoint = write_checkpoint(tmp_path / "step1", values, "model.{name}")
    hf_dir = write_export(tmp_path / "hf", values, "bf16-grid")
    code, result = run(monkeypatch, checkpoint, hf_dir, "fp32", tmp_path / "result.json")
    assert code == 1 and not result["ok"]
    assert len(result["failures"]) == 10  # decay: float32 but rounded; convolutions: bfloat16


@pytest.mark.parametrize(
    "kda_decay,name,edit,problem",
    [
        ("fp32", "model.layers.2.linear_attn.k_conv1d.weight", lambda t: t.to(torch.bfloat16), "stored as"),
        ("fp32", "model.layers.0.linear_attn.A_log", lambda t: torch.nextafter(t, t + 1), "elements differ"),
        ("bf16-grid", "model.layers.2.linear_attn.dt_bias", lambda t: t.to(torch.bfloat16), "stored as"),
        ("bf16-grid", "model.layers.0.linear_attn.v_conv1d.weight", lambda t: t * 2, "elements differ"),
        ("fp32", "model.layers.0.linear_attn.dt_bias", lambda t: t[:2], "elements, the checkpoint has"),
    ],
)
def test_one_wrong_tensor_fails_and_is_named(tmp_path, monkeypatch, kda_decay, name, edit, problem):
    values = masters()
    checkpoint = write_checkpoint(tmp_path / "step1", values, "model.{name}")
    hf_dir = write_export(tmp_path / "hf", values, kda_decay, change=(name, edit))
    code, result = run(monkeypatch, checkpoint, hf_dir, kda_decay, tmp_path / "result.json")
    assert code == 1
    (failure,) = result["failures"]
    assert name == f"model.layers.{failure['layer']}.linear_attn.{failure['weight']}"
    assert problem in failure["problem"]


def test_missing_tensor_is_an_error(tmp_path):
    values = masters()
    hf_dir = write_export(tmp_path / "hf", {k: v for k, v in values.items() if k != (2, "A_log")}, "fp32")
    with pytest.raises(KeyError, match="A_log"):
        check_script.read_export(hf_dir, list(KDA_LAYERS))
    checkpoint = write_checkpoint(
        tmp_path / "step1", {k: v for k, v in values.items() if k != (0, "dt_bias")}, "model.{name}"
    )
    with pytest.raises(KeyError, match="blocks.0.attention.dt_bias"):
        check_script.read_checkpoint(checkpoint, list(KDA_LAYERS))
