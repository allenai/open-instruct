"""Actual torch module hooks preserve outputs and capture only an exact armed prefix."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from scripts.miles import update_zero_capture as capture


class TopK(torch.nn.Module):
    def forward(self, hidden_states, router_logits):
        values, indices = router_logits.topk(2, dim=-1)
        return SimpleNamespace(topk_ids=indices, topk_weights=values.softmax(-1))


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 5, bias=False)
        self.topk = TopK()

    def forward(self, x):
        routes = self.topk(x, self.gate(x))
        return x * routes.topk_weights.sum(-1, keepdim=True)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4)
        self.mlp = MLP()

    def forward(self, positions, hidden_states, forward_batch=None):
        return hidden_states + self.mlp(self.self_attn(hidden_states))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(9, 4)
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.model.norm = torch.nn.LayerNorm(4)

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(None, x)
        return SimpleNamespace(next_token_logits=self.model.norm(x)[-1:])


def test_actual_hooks_exact_input_two_phases_and_no_output_changes(tmp_path):
    ids = [1, 2, 3, 4] * 80
    model = Model()
    expected = model(torch.tensor(ids)).next_token_logits.clone()
    capture.attach_capture(model, tmp_path)
    capture.attach_capture(model, tmp_path)  # idempotent
    model(torch.tensor(ids))  # unarmed startup
    assert not list(tmp_path.rglob("*.pt"))
    request = capture.arm_capture(tmp_path, "hf", "case341", "hf341", ids)
    assert request["sampling_params"]["max_new_tokens"] == 1
    model(torch.tensor(ids[:-1]))  # wrong length, chunk/prefix reuse must not qualify
    model(torch.tensor([2, *ids[1:]]))  # same length, wrong teacher-forced tokens
    assert not list(tmp_path.rglob("*.pt"))
    for phase in ("hf", "published"):
        capture.arm_capture(tmp_path, phase, "case341", phase + "341", ids)
        actual = model(torch.tensor(ids)).next_token_logits
        assert torch.equal(actual, expected)
        path = tmp_path / f"worker-{os.getpid()}" / f"{phase}341.pt"
        record = torch.load(path, weights_only=True)
        assert record["input_ids"] == ids
        assert record["positions"] == [*range(16), *range(192, 320)]
        assert record["activations"]["model.layers.0.input"].shape == (144, 4)
        assert record["routes"]["model.layers.0.mlp.topk"]["logits"].shape == (320, 5)
        assert record["routes"]["model.layers.0.mlp.topk"]["topk_ids"].shape == (320, 2)
        assert record["routes"]["model.layers.0.mlp.topk"]["canonical_set_matches"].all()
        assert record["sources"]
        proof = json.loads(path.with_suffix(".json").read_text())
        assert proof["sha256"] == capture.digest(path.read_bytes())
        original = path.read_bytes()
        model(torch.tensor(ids))  # same marker never overwrites retained evidence
        assert path.read_bytes() == original


def test_rowwise_router_margin_and_storage_dtype():
    logits = torch.tensor([[1, 3, 2, 0], [4, 4, 1, 0]], dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2], [1, 0]], dtype=torch.int32)
    record = capture.route_record(logits, ids, torch.ones(2, 2) / 2)
    assert record["logits_dtype"] == "torch.bfloat16"
    assert record["boundary_margin"].tolist() == [1, 3]
    assert record["canonical_set_matches"].tolist() == [True, True]
    ids[1] = torch.tensor([1, 2])
    assert not capture.route_record(logits, ids, torch.ones(2, 2) / 2)["canonical_set_matches"][1]


@pytest.mark.parametrize(
    "change",
    [
        {"capture_id": "../x"},
        {"case_id": ""},
        {"phase": "x/y"},
        {"input_ids": [True]},
        {"positions": [1, 1]},
        {"positions": [-1]},
    ],
)
def test_invalid_marker_rejected(change):
    arm = {"phase": "hf", "case_id": "341", "capture_id": "hf341", "input_ids": [1, 2], "positions": [0, 1]}
    with pytest.raises(ValueError):
        capture.validate_arm({**arm, **change})


def test_marker_without_positions_matches_driver_contract(tmp_path):
    model = Model()
    capture.attach_capture(model, tmp_path)
    (tmp_path / "capture-request.json").write_text(
        json.dumps({"phase": "hf", "case_id": "341", "capture_id": "hf341", "input_ids": [1, 2]})
    )
    model(torch.tensor([1, 2]))
    assert len(list(tmp_path.rglob("*.pt"))) == 1


def test_import_hook_patches_constructor_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.TRACE_ENV, str(tmp_path))

    class External(Model):
        pass

    module = SimpleNamespace(Olmo3MoeForCausalLM=External)
    capture._patch_model(module)
    capture._patch_model(module)
    model = External()
    assert model._oi_prefill_capture_installed
    capture.arm_capture(tmp_path, "hf", "341", "hf341", [1, 2])
    model(torch.tensor([1, 2]))
    assert len(list(tmp_path.rglob("*.pt"))) == 1


def test_autotune_snapshot_unwraps_deduplicates_and_never_invokes(monkeypatch):
    class Tuner:
        def __init__(self, cache):
            self.cache = cache

        def __call__(self):
            pytest.fail("Reading choices must never invoke kernels")

    config = SimpleNamespace(kwargs={"BLOCK_M": 64}, num_warps=4, num_stages=2, num_ctas=1, maxnreg=128)
    tuner = Tuner({(177, "bf16"): config})
    module = ModuleType("fla.test_update_zero_snapshot")
    module.a = SimpleNamespace(fn=SimpleNamespace(fn=tuner))
    module.b = tuner
    module.empty = Tuner({})
    unrelated = ModuleType("unrelated_update_zero_snapshot")
    unrelated.tuner = Tuner({(1,): config})
    monkeypatch.setattr(capture, "Autotuner", Tuner)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    snapshot = capture.snapshot_autotune_configs()
    assert snapshot == {
        "fla.test_update_zero_snapshot.a": {
            "(177, 'bf16')": {
                "kwargs": {"BLOCK_M": 64},
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
                "maxnreg": 128,
            }
        }
    }


def test_autotune_policy_records_controls_without_unrelated_secrets(monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/cache")
    monkeypatch.setenv("UNRELATED_SECRET", "not-recorded")
    monkeypatch.setitem(
        sys.modules, "fla.ops.utils.cache", SimpleNamespace(FLA_CACHE_MODE=SimpleNamespace(value="memory"))
    )
    monkeypatch.setitem(sys.modules, "fla.utils._config", SimpleNamespace(FLA_CACHE_RESULTS=True))
    policy = capture.autotune_policy()
    assert policy["cache_mode"] == "memory" and policy["cache_results"] is True
    assert policy["environment"]["TRITON_CACHE_DIR"] == "/tmp/cache"
    assert "UNRELATED_SECRET" not in policy["environment"]
