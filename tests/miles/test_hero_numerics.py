"""The diagnostic must observe inputs before an in-place serving kernel changes them."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import torch as safetensors_torch
from scripts.miles import hero_expert_rounding, hero_numerics
from torch import nn


class InPlace(nn.Module):
    def forward(self, value):
        return value.add_(1)


def test_trace_preserves_pre_kernel_input_and_distinct_module_names(tmp_path):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"case": "prime"}))
    modules = {}
    for name in ["model.layers.1.self_attn", "model.layers.1.mlp"]:
        module = InPlace()
        module.register_forward_hook(hero_numerics.trace_factory({"root": str(tmp_path), "module": name}))
        module(torch.zeros(2, 3))
        modules[name] = module
    assert not (tmp_path / "sglang").exists()
    control.write_text(json.dumps({"case": "16"}))
    for module in modules.values():
        module(torch.zeros(2, 3))
    files = list((tmp_path / "sglang" / "16").glob("*.pt"))
    assert {p.stem for p in files} == set(modules)
    for path in files:
        saved = torch.load(path, weights_only=True)
        torch.testing.assert_close(saved["input"], torch.zeros(2, 3))
        torch.testing.assert_close(saved["output"], torch.ones(2, 3))


class Logits(nn.Module):
    def forward(self, hidden):
        return SimpleNamespace(next_token_logits=hidden + 2)


class DirectForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_processor = Logits()

    def forward(self, hidden):
        return self.logits_processor(hidden)


def test_logits_capture_when_runner_calls_model_forward_directly(tmp_path):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"case": "prime"}))
    model = DirectForward()
    model.logits_processor.register_forward_hook(
        hero_numerics.trace_factory({"root": str(tmp_path), "module": "logits_processor"})
    )
    model.forward(torch.zeros(1, 3))
    control.write_text(json.dumps({"case": "16"}))
    model.forward(torch.zeros(1, 3))
    saved = torch.load(tmp_path / "sglang" / "16" / "logits.pt", weights_only=True)
    torch.testing.assert_close(saved["output"], torch.full((1, 3), 2.0))


@pytest.mark.parametrize("sharded", [False, True])
def test_expert_reader_supports_single_file_and_sharded_exports(tmp_path, monkeypatch, sharded):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    weights = {
        f"model.layers.1.mlp.experts.{i}.{projection}_proj.weight": torch.full(shape, i * 3 + value)
        for i in range(2)
        for projection, shape, value in [("gate", (4, 2), 1.0), ("up", (4, 2), 2.0), ("down", (2, 4), 3.0)]
    }
    if sharded:
        keys = list(weights)
        index = {}
        for i, subset in enumerate([keys[:3], keys[3:]]):
            filename = f"model-{i:05d}.safetensors"
            safetensors_torch.save_file({key: weights[key] for key in subset}, tmp_path / filename)
            index.update({key: filename for key in subset})
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        safetensors_torch.save_file(weights, tmp_path / "model.safetensors")
    gate_up, down = hero_expert_rounding.expert_weights(tmp_path, 1, 2)
    assert gate_up.shape == (2, 2, 8)
    assert down.shape == (2, 4, 2)
    torch.testing.assert_close(gate_up[1, :, :4], torch.full((2, 4), 4.0))
    torch.testing.assert_close(gate_up[1, :, 4:], torch.full((2, 4), 5.0))
    torch.testing.assert_close(down[1], torch.full((4, 2), 6.0))
