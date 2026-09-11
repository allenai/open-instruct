"""Check explicit token alignment, router axes, observer neutrality, and cleanup."""

import copy
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import update_zero_training_capture as capture
from torch import nn


class Router(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.weight = nn.Parameter(torch.randn(4, 19))

    def gating(self, value):
        return value @ self.weight.T

    def forward(self, value):
        logits = self.gating(value)
        scores = logits.softmax(-1)
        weights, ids = scores.topk(2, dim=-1)
        if self.backend == "olmo_core":
            return weights, ids, None, (scores, logits, None, None, 1)
        probabilities = torch.zeros_like(scores).scatter(-1, ids, weights)
        mapping = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, ids, True)
        return probabilities.reshape(-1, 4), mapping.reshape(-1, 4)


class Block(nn.Module):
    def __init__(self, backend):
        super().__init__()
        if backend == "olmo_core":
            self.routed_experts_router = Router(backend)
        else:
            self.mlp = nn.Module()
            self.mlp.router = Router(backend)

    def forward(self, value):
        router = self.routed_experts_router if hasattr(self, "routed_experts_router") else self.mlp.router
        router(value)
        return value * 1.25


class Model(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.embedding = nn.Embedding(19, 19)
        self.backend = backend
        if backend == "olmo_core":
            self.blocks = nn.ModuleList([Block(backend)])
        else:
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([Block(backend)])

    def forward(self, input_ids):
        value = self.embedding(input_ids)
        blocks = self.blocks if self.backend == "olmo_core" else self.decoder.layers
        return blocks[0](value)


def cases():
    return [
        dict(case_id="a", input_ids=[1, 2, 5, 8], response_length=3, loss_mask=[1, 0, 1]),
        dict(case_id="b", input_ids=[4, 3, 1, 6, 7, 2], response_length=2, loss_mask=[0, 1]),
    ]


def run_scores(model, samples):
    outputs = []
    with torch.no_grad():
        for case in samples:
            tokens = case["input_ids"]
            padded = tokens + [0] * (6 - len(tokens)) if model.backend == "megatron" else tokens
            logits = model(torch.tensor([padded]))[0]
            start = len(tokens) - case["response_length"]
            outputs.append(torch.stack([logits[i - 1].log_softmax(-1)[tokens[i]] for i in range(start, len(tokens))]))
    return outputs


@pytest.mark.parametrize("backend", ["olmo_core", "megatron"])
def test_real_module_hooks_preserve_scores_and_canonical_token_layer_axes(backend):
    torch.manual_seed(17)
    model, samples = Model(backend), cases()
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    before = run_scores(model, samples)
    with capture.Recorder(model, samples, backend) as recorder:
        after = run_scores(model, samples)
    assert recorder.errors == []
    recorder.validate()
    capture.assert_scores(recorder.records, samples, before, after)
    for case, record in zip(samples, recorder.records, strict=True):
        route = record["routes"][0]
        assert route["topk_ids"].shape == (len(case["input_ids"]), 2)
        assert route["logits"].shape == (len(case["input_ids"]), 4)
        assert record["target_token_positions"] == list(
            range(len(case["input_ids"]) - case["response_length"], len(case["input_ids"]))
        )
        assert record["loss_mask"] == case["loss_mask"]
        assert torch.equal(route["topk_ids"], route["topk_ids"].sort(-1).values)
    assert all(torch.equal(v, parameters[k]) for k, v in model.state_dict().items())
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_wrong_next_token_shift_is_rejected():
    model, samples = Model("olmo_core"), cases()
    before = run_scores(model, samples)
    with capture.Recorder(model, samples, "olmo_core") as recorder:
        after = run_scores(model, samples)
    recorder.records[0]["reference_log_probs"] += 0.25
    with pytest.raises(ValueError, match="logits"):
        capture.assert_scores(recorder.records, samples, before, after)


def test_observer_errors_are_deferred_until_after_native_forward():
    model, samples = Model("megatron"), cases()
    recorder = capture.Recorder(model, samples, "megatron")
    original = model.decoder.layers[0].mlp.router.gating
    with recorder:
        recorder.route = lambda *args: (_ for _ in ()).throw(ValueError("bad capture"))
        outputs = run_scores(model, samples)
    assert len(outputs) == 2
    assert recorder.errors == ["ValueError: bad capture"] * 2
    assert model.decoder.layers[0].mlp.router.gating == original
    assert not model._forward_hooks


def test_prefix_and_retained_rollout_payloads_preserve_exact_tokens_and_masks():
    samples = cases()
    prefix = capture.prefix_payload({"cases": samples})
    assert prefix["cases"][0]["input_ids"] == samples[0]["input_ids"]
    assert prefix["cases"][0]["response_length"] == 3
    originals = [
        dict(tokens=x["input_ids"], response_length=x["response_length"], loss_mask=x["loss_mask"]) for x in samples
    ]
    frozen = copy.deepcopy(originals)
    payload = capture.rollout_payload(originals)
    assert originals == frozen
    assert payload["cases"][1]["loss_mask"] == [0, 1]
    assert capture.validate_payload(payload, 2, 6) == payload["cases"]
    assert [x["case_id"] for x in payload["cases"][0::2]] == ["rollout0-sample0"]
    with pytest.raises(ValueError, match="divide"):
        capture.validate_payload(payload, 3, 6)
    payload["cases"][0]["loss_mask"] = [1]
    with pytest.raises(ValueError, match="mask"):
        capture.validate_payload(payload, 2, 6)


def test_observer_score_change_is_rejected():
    samples = cases()
    before = [torch.zeros(x["response_length"]) for x in samples]
    after = [x.clone() for x in before]
    after[0][0] = 1
    with pytest.raises(ValueError, match="changed"):
        capture.assert_scores([{}, {}], samples, before, after)


def test_layer_mapping_and_bounded_positions():
    assert capture.layer_id("module.blocks.19.routed_experts_router", "olmo_core") == 19
    assert capture.layer_id("module.module.decoder.layers.19.mlp.router", "megatron") == 19
    assert capture.positions(619) == list(range(16)) + list(range(491, 619))
    with pytest.raises(ValueError, match="map"):
        capture.layer_id("unrecognized.router", "megatron")


def test_worker_setup_preserves_original_hook_before_registration(monkeypatch):
    events = []
    monkeypatch.setenv("OI_TRAINER_ROUTE_ORIGINAL_WORKER_HOOK", "original.worker.setup")
    monkeypatch.setenv("OI_TRAINER_ROUTE_BACKEND", "megatron")
    monkeypatch.setattr(
        capture.importlib, "import_module", lambda name: SimpleNamespace(setup=lambda: events.append(name))
    )
    monkeypatch.setattr(capture, "install", lambda backend: events.append(backend))
    capture.worker_setup()
    assert events == ["original.worker", "megatron"]


def test_missing_router_layer_is_rejected_after_scoring():
    model, samples = Model("olmo_core"), cases()
    with capture.Recorder(model, samples, "olmo_core") as recorder:
        run_scores(model, samples)
    del recorder.records[0]["routes"][0]
    with pytest.raises(ValueError, match="omitted"):
        recorder.validate()
