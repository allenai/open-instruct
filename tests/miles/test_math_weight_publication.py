"""Offline guards for the study's live transport gate; GPU evidence is separate."""

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import diagnose_math_nvfp4_encoding as encoder
from scripts.miles import qualify_math_weight_publication as gate


def scores(delta=0.0):
    return [{"meta_info": {"input_token_logprobs": [[None, 0]] + [[-2.0 + delta, i] for i in range(1, 40)]}}]


def test_publication_context_matches_immutable_runtime_loader():
    path = Path("/opt/core-rl/sources/sglang/python/sglang/srt/model_loader/loader.py")
    tree = ast.parse(path.read_text())
    loader = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DefaultModelLoader"
    )
    method = next(
        node for node in loader.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights_only"
    )
    scopes = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "temp_set_env"
    ]
    assert len(scopes) == 1
    expected = {keyword.arg: ast.literal_eval(keyword.value) for keyword in scopes[0].keywords}
    assert gate.quantized_publication.NVFP4_LOADER_ENV == encoder.LOADER_ENV == expected


def test_encoder_replay_uses_exact_pinned_loader_scope_and_restores_on_failure(monkeypatch):
    contexts = []

    class Scope:
        def __enter__(self):
            contexts.append("enter")

        def __exit__(self, *error):
            contexts.append(("exit", error[0]))

    def environment(**values):
        assert values == encoder.LOADER_ENV
        assert values["FLASHINFER_NVFP4_4OVER6"] == "1"
        assert values["FLASHINFER_NVFP4_4OVER6_ERR_MODE"] == "MSE"
        return Scope()

    monkeypatch.setattr(encoder.importlib, "import_module", lambda _: SimpleNamespace(temp_set_env=environment))
    with encoder.encoding_context("ambient"):
        pass
    assert contexts == []
    with pytest.raises(RuntimeError, match="native error"), encoder.encoding_context("pinned_loader"):
        raise RuntimeError("native error")
    assert contexts == ["enter", ("exit", RuntimeError)]
    with pytest.raises(ValueError, match="Unknown encoding context"), encoder.encoding_context("invalid"):
        pass


def test_dispatch_snapshot_retains_selected_configs_without_mutating_cache(monkeypatch):
    cache = {(43, "bf16"): "native config"}
    kernel = SimpleNamespace(cache=cache, configs=["native config"], best_config="native config")
    monkeypatch.setitem(sys.modules, "fla.test_publication_dispatch", SimpleNamespace(kernel=kernel))
    result = gate.fla_dispatch_snapshot()["fla.test_publication_dispatch.kernel"]
    assert result["cache"] == {"(43, 'bf16')": "'native config'"}
    assert result["best_config"] == "'native config'"
    assert cache == {(43, "bf16"): "native config"}


def test_dispatch_snapshot_unwraps_heuristics_and_serializes_launch_choices(monkeypatch):
    config = SimpleNamespace(kwargs={"BK": 128}, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None)
    cache = {(43, "bf16"): config}
    kernel = SimpleNamespace(cache=cache, configs=[config], best_config=config)
    wrapper = SimpleNamespace(fn=kernel)
    monkeypatch.setitem(sys.modules, "olmo_sglang.kda.test_dispatch", SimpleNamespace(kernel=wrapper))
    result = gate.fla_dispatch_snapshot()["olmo_sglang.kda.test_dispatch.kernel.fn"]
    assert result["cache"]["(43, 'bf16')"] == result["best_config"]
    assert result["best_config"]["kwargs"] == {"BK": 128}
    assert result["best_config"]["num_warps"] == 4
    assert result["best_config"]["num_stages"] == 2
    assert json.loads(json.dumps(result)) == result
    assert wrapper.fn is kernel and cache[(43, "bf16")] is config


@pytest.mark.parametrize("capture_index", [0, 1])
def test_kda_diagnostic_captures_prior_state_without_replaying_or_changing_result(
    monkeypatch, tmp_path, capture_index
):
    calls = []

    class Kernel:
        def _gather_initial_state(self, states, indices):
            return states.index_select(0, indices), indices >= 0

        def _run(self, *values, **options):
            calls.append(values)
            options["ssm_states"].add_(1)
            return values[0], "native metadata"

    monkeypatch.setenv("OI_PUBLICATION_TRACE_KDA", "1")
    monkeypatch.setenv("OI_PUBLICATION_KDA_CAPTURE_INDEX", str(capture_index))
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    monkeypatch.setattr(gate.importlib, "import_module", lambda _: SimpleNamespace(OlmoFLAKDAKernel=Kernel))
    gate.instrument_kda_inputs()
    values = [torch.ones(1, 43, 2, 4) for _ in range(5)]
    state = torch.zeros(3, 2, 4, 4)
    options = {"ssm_states": state, "cache_indices": torch.tensor([1]), "track_state": None}
    for _ in range(3):
        result = Kernel()._run(*values, **options)
        assert result[0] is values[0] and result[1] == "native metadata"
    assert len(calls) == 3 and torch.equal(state, torch.full_like(state, 3))
    saved = torch.load(next(tmp_path.glob("kda-inputs-*.pt")), weights_only=True)
    assert torch.equal(saved["initial_state"], torch.full_like(saved["initial_state"], capture_index))
    receipt = json.loads(next(tmp_path.glob("kda-receipt-*.json")).read_text())
    assert receipt["prefill_recurrence_index"] == capture_index
    assert "ssm_states" not in saved["options"] and "track_state" not in saved["options"]
    assert len(list(tmp_path.glob("kda-inputs-*.pt"))) == 1
    assert torch.equal(torch.load(next(tmp_path.glob("kda-output-*.pt")), weights_only=True), values[0])


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
    monkeypatch.delenv("OI_PUBLICATION_TRACE_ENCODING", raising=False)
    monkeypatch.delenv("OI_PUBLICATION_TRACE_PACKING", raising=False)

    def unexpected_instrumentation():
        raise AssertionError("Strict gate must not synchronize inside quantization/packing")

    monkeypatch.setattr(gate, "instrument_receiver", lambda: None)
    monkeypatch.setattr(gate, "instrument_nvfp4_processing", unexpected_instrumentation)
    monkeypatch.setattr(gate, "instrument_nvfp4_quantization", unexpected_instrumentation)
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


def test_audit_includes_norm_parameters_without_quant_methods(monkeypatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(2)

        def load_weights(self, weights):
            return set()

        def forward(self):
            return "native output"

    model = Model()
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    monkeypatch.setattr(gate, "instrument_receiver", lambda: None)
    monkeypatch.setattr(gate, "instrument_nvfp4_processing", lambda: None)
    monkeypatch.setattr(gate, "instrument_nvfp4_quantization", lambda: None)
    modules = {
        "olmo_sglang.models.olmo3_moe": SimpleNamespace(Olmo3MoeForCausalLM=Model),
        "sglang.srt.managers.scheduler": SimpleNamespace(run_scheduler_process=lambda: model.forward()),
    }
    monkeypatch.setattr(gate.importlib, "import_module", modules.__getitem__)
    assert gate.audit_scheduler() == "native output"
    rows = json.loads(next(tmp_path.glob("audit-*.json")).read_text())
    assert len(rows) == 1 and rows[0]["name"] == "norm"
    assert rows[0]["class"] == "LayerNorm" and rows[0]["quant_method"] == "NoneType"
    assert rows[0]["parameters"]["weight"] == gate.tensor_fingerprint(model.norm.weight)


def test_encoding_diagnostics_preserve_native_arguments_results_and_sampling(monkeypatch, tmp_path):
    weight = torch.arange(16.0).reshape(4, 4)
    scale = torch.tensor(0.25)
    encoded = (weight.to(torch.uint8), weight.clone(), scale)
    calls = []

    class Quantizer:
        @staticmethod
        def _quantize_weight_nvfp4(actual, actual_scale=None):
            assert actual is weight and actual_scale is scale
            calls.append(True)
            return encoded

    monkeypatch.setattr(
        gate.importlib, "import_module", lambda _: SimpleNamespace(ModelOptNvFp4OnlineFusedMoEMethod=Quantizer)
    )
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    before = weight.clone()
    gate.instrument_nvfp4_quantization()
    for _ in range(1025):
        assert Quantizer._quantize_weight_nvfp4(weight, scale) is encoded
    receipts = [json.loads(path.read_text()) for path in tmp_path.glob("encoding-*.json")]
    assert len(calls) == 1025
    assert {receipt["call"] for receipt in receipts} == {0, 1, 2, 3, 1024}
    for receipt in receipts:
        assert receipt["input"] == gate.tensor_fingerprint(weight)
        assert receipt["encoded"] == [gate.tensor_fingerprint(value) for value in encoded]
        assert receipt["provided_scale"] == gate.tensor_fingerprint(scale)
        assert receipt["input_stride"] == [4, 1]
    assert torch.equal(weight, before)


def test_prefill_trace_preserves_outputs_and_removes_hooks(monkeypatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False)
            self.register_buffer("derived", torch.tensor([7.0]))

        def forward(self, value):
            return self.projection(value)

    model = Model()
    value = torch.ones(3, 2)
    expected = model(value)
    monkeypatch.setenv("OI_PUBLICATION_AUDIT", str(tmp_path))
    actual = gate.trace_forward(model, Model.forward, (value,), {})
    assert torch.equal(actual, expected)
    assert model._publication_prefill_traced
    assert not model.projection._forward_hooks
    receipt = json.loads(next(tmp_path.glob("prefill-trace-*.json")).read_text())
    assert receipt["diagnostic_only"] and receipt["synchronizes_cuda"]
    assert receipt["buffers"]["derived"]["sha256"] == gate.tensor_fingerprint(model.derived)["sha256"]
    assert receipt["modules"][0]["module"] == "projection"
    assert receipt["modules"][0]["output"]["sha256"] == gate.tensor_fingerprint(expected)["sha256"]

    def broken(model, value):
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        gate.trace_forward(model, broken, (value,), {})
    assert not model.projection._forward_hooks
