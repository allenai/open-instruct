"""Offline guards for the study's live transport gate; GPU evidence is separate."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import qualify_math_weight_publication as gate


def scores(delta=0.0):
    return [{"meta_info": {"input_token_logprobs": [[None, 0]] + [[-2.0 + delta, i] for i in range(1, 40)]}}]


def test_sender_matches_pinned_trainer_cumem_default_and_preserves_override(monkeypatch):
    monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
    gate.configure_transport_environment()
    assert gate.os.environ["NCCL_CUMEM_ENABLE"] == "0"
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    gate.configure_transport_environment()
    assert gate.os.environ["NCCL_CUMEM_ENABLE"] == "1"


def test_receiver_preserves_causal_error_and_context(monkeypatch, tmp_path):
    class Receiver:
        gpu_id, device = 2, "cuda"

        def receive_weights_from_distributed(self, **options):
            raise RuntimeError("causal collective failure")

    monkeypatch.setattr(gate.importlib, "import_module", lambda _: SimpleNamespace(WeightUpdater=Receiver))
    monkeypatch.setattr(gate.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
    gate.instrument_receiver()
    with pytest.raises(RuntimeError, match="causal collective failure"):
        Receiver().receive_weights_from_distributed(names=["expert.weight"])
    context = json.loads((tmp_path / "transport-context.json").read_text())
    error = json.loads((tmp_path / "transport-error.json").read_text())
    assert context["current_device"] == context["gpu_id"] == 2
    assert context["NCCL_CUMEM_ENABLE"] == "0"
    assert "causal collective failure" in error["traceback"]


def test_packing_diagnostics_observe_two_experts_without_changing_processing(monkeypatch, tmp_path):
    class Quantizer:
        def process_weights_after_loading(self, layer):
            layer.weight.data.add_(1)
            return "native result"

    monkeypatch.setattr(
        gate.importlib, "import_module", lambda _: SimpleNamespace(ModelOptNvFp4OnlineFusedMoEMethod=Quantizer)
    )
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.arange(12.0).reshape(3, 4)))
    layer.register_parameter("scale", torch.nn.Parameter(torch.tensor(1.0)))
    before = layer.weight.detach().clone()
    gate.instrument_nvfp4_processing()
    for epoch in (1, 2):
        assert Quantizer().process_weights_after_loading(layer) == "native result"
        assert torch.equal(layer.weight, before + epoch)
        receipt = json.loads(next(tmp_path.glob(f"packing-*-{epoch}.json")).read_text())
        assert receipt["epoch"] == epoch and receipt["sample_experts"] == [0, 1]
        assert receipt["before"]["weight"] == gate.tensor_fingerprint((before + epoch - 1)[:2])
        assert receipt["after"]["weight"] == gate.tensor_fingerprint((before + epoch)[:2])
        assert receipt["before"]["scale"] == receipt["after"]["scale"]


def test_load_diagnostics_keep_iterator_items_and_report_stale_cache(monkeypatch, tmp_path):
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    weight = torch.ones(2, 2, dtype=torch.bfloat16)
    item = (name, weight, {"sentinel": True})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._weight_targets = SimpleNamespace(params={"stale": object()})

        def load_weights(self, weights):
            received = list(weights)
            assert len(received) == 1 and received[0] is item
            return {name}

        def forward(self):
            pass

    model = Model()
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    monkeypatch.setattr(gate, "instrument_receiver", lambda: None)
    monkeypatch.setattr(gate, "instrument_nvfp4_processing", lambda: None)
    modules = {
        "olmo_sglang.models.olmo3_moe": SimpleNamespace(Olmo3MoeForCausalLM=Model),
        "sglang.srt.managers.scheduler": SimpleNamespace(
            run_scheduler_process=lambda: model.load_weights(iter([item]))
        ),
    }
    monkeypatch.setattr(gate.importlib, "import_module", modules.__getitem__)
    assert gate.audit_scheduler() == {name}
    receipt = json.loads(next(tmp_path.glob("incoming-*.json")).read_text())
    assert receipt["weights"] == {name: gate.tensor_fingerprint(weight)}
    assert receipt["stale_cached_parameters"] == ["stale"]
    assert torch.equal(weight, torch.ones_like(weight))


def test_reuse_requires_matching_mutation_hash_and_model(tmp_path):
    checkpoint = tmp_path / "changed-checkpoint-1"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"preserved qualification checkpoint")
    receipt = {
        "source": str(tmp_path / "original"),
        "variant": 1,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (checkpoint / "qualification-mutation.json").write_text(json.dumps(receipt))
    output = tmp_path / "new-attempt"
    output.mkdir()
    args = SimpleNamespace(reuse_mutation_checkpoints=tmp_path, model=tmp_path / "original", output=output)
    assert gate.fresh_checkpoint(args, 1) == checkpoint
    assert json.loads((output / "reused-mutation-1.json").read_text())["sha256"] == receipt["sha256"]
    weights.write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.fresh_checkpoint(args, 1)
    args.model = tmp_path / "other-model"
    with pytest.raises(ValueError, match="provenance mismatch"):
        gate.fresh_checkpoint(args, 1)


def test_fixed_token_gate_rejects_stale_alignment_nonfinite_and_drift():
    reference = scores()
    assert gate.compare_scores(reference, reference, "bf16", unchanged=True)["max_abs"] == 0
    with pytest.raises(ValueError, match="mismatch"):
        gate.compare_scores(scores(1e-5), reference, "bf16", unchanged=True)
    assert gate.compare_scores(scores(1e-5), reference, "bf16")["tokens"] == 39
    with pytest.raises(ValueError, match="mismatch"):
        gate.compare_scores(scores(0.01), reference, "bf16")
    for mutation in ("nan", "tokens", "missing", "null"):
        actual = copy.deepcopy(reference)
        if mutation == "nan":
            actual[0]["meta_info"]["input_token_logprobs"][4][0] = float("nan")
        elif mutation == "tokens":
            actual[0]["meta_info"]["input_token_logprobs"][4][1] = 999
        elif mutation == "missing":
            actual[0]["meta_info"]["input_token_logprobs"].pop()
        else:
            actual[0]["meta_info"]["input_token_logprobs"][4][0] = None
        with pytest.raises(ValueError):
            gate.compare_scores(actual, reference, "bf16")


def test_buckets_preserve_every_name_and_force_gate_up_split_across_layers():
    state = {"lm_head.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    for layer in (1, 8, 15):
        for expert in (0, 1, 511):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                state[f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"] = torch.ones(
                    2, 2, dtype=torch.bfloat16
                )
    state["model.layers.1.mlp.shared_expert.up_proj.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)

    class Reader:
        def keys(self):
            return state.keys()

        def get_tensor(self, name):
            return state[name]

    for variant in (0, 1, 2):
        buckets = list(gate.weight_buckets(Reader(), variant, 32))
        flattened = [(n, t) for bucket in buckets for n, t in bucket]
        assert sorted(n for n, _ in flattened) == sorted(state)
        assert sum(len(bucket) for bucket in buckets) == len(state)
        for bucket in buckets:
            assert sum(t.nbytes for _, t in bucket) <= 32
            for name, tensor in bucket:
                match = gate.EXPERT.fullmatch(name)
                if match and match[2] == "0":
                    assert len(bucket) == 1
                expected = 1.0 if not match or variant == 0 else (0.875, 1.125)[variant - 1]
                assert torch.equal(tensor, torch.full_like(tensor, expected))
        assert all(torch.equal(t, torch.ones_like(t)) for t in state.values())
