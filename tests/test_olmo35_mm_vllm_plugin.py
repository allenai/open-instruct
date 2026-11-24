# Ported from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/tests/test_vllm_multimodal.py
# (plugin path re-pointed at scripts/export/olmo35_mm/vllm_plugin). CPU-only except the last test, which needs vLLM
# and the transformers plugin installed (skipped otherwise).
"""Language-model-only serving of whole-model (vision + OLMoE3) exports in the vLLM plugin."""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import pytest

# Imports of the plugin stay inside the tests: each test imports it fresh from source (with fake vllm /
# transformers-plugin modules where needed) and checks what registration does and does not import.
PLUGIN = Path(__file__).resolve().parents[1] / "scripts" / "export" / "olmo35_mm" / "vllm_plugin"
VISION_ARCHITECTURE = "Olmo3MoeVisionForConditionalGeneration"


@pytest.fixture
def plugin_package(monkeypatch):
    """Import the plugin package from source; its ``__init__`` imports neither torch nor vLLM."""
    monkeypatch.syspath_prepend(str(PLUGIN / "src"))
    for name in list(sys.modules):
        if name == "olmoe3_vllm_plugin" or name.startswith("olmoe3_vllm_plugin."):
            monkeypatch.delitem(sys.modules, name)
    import olmoe3_vllm_plugin  # noqa: PLC0415

    return olmoe3_vllm_plugin


# --- pure-Python tensor-name routing -------------------------------------------------------


def test_stream_strips_language_model_prefix_and_counts_vision_tensors(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import MultimodalWeightStream  # noqa: PLC0415

    # Interleaved file order, as safetensors iterates a merged export.
    weights = [
        ("language_model.model.embed_tokens.weight", 1),
        ("vision_tower.patch_embedding.weight", 2),
        ("language_model.model.layers.0.self_attn.q_proj.weight", 3),
        ("connector.pooling.wq.weight", 4),
        ("vision_tower.blocks.0.attn.wq.bias", 5),
        ("language_model.lm_head.weight", 6),
    ]
    stream = MultimodalWeightStream(iter(weights))
    assert list(stream) == [
        ("model.embed_tokens.weight", 1),
        ("model.layers.0.self_attn.q_proj.weight", 3),
        ("lm_head.weight", 6),
    ]
    assert stream.language_model_tensors == 3
    assert stream.skipped == {"vision_tower.": 2, "connector.": 1}
    assert stream.summary() == (
        "3 language_model.* tensor(s) read, skipped 3 vision tensor(s) (2 vision_tower.*, 1 connector.*)"
    )


def test_stream_is_lazy_so_the_text_loader_streams_tensors(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import MultimodalWeightStream  # noqa: PLC0415

    pulled: list[str] = []

    def source():
        for name in ("language_model.a", "vision_tower.b", "language_model.c"):
            pulled.append(name)
            yield name, None

    stream = MultimodalWeightStream(source())
    iterator = iter(stream)
    assert pulled == []
    assert next(iterator) == ("a", None)
    assert pulled == ["language_model.a"]
    assert next(iterator) == ("c", None)
    assert pulled == ["language_model.a", "vision_tower.b", "language_model.c"]
    assert stream.skipped["vision_tower."] == 1


def test_stream_rejects_tensors_outside_the_export_contract(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import MultimodalWeightStream  # noqa: PLC0415

    # A text export's bare names are not a multimodal export.
    with pytest.raises(KeyError, match="Unexpected Olmo3MoeVision checkpoint weight: 'model.embed"):
        list(MultimodalWeightStream([("model.embed_tokens.weight", 0)]))
    # Nor is a prefix that merely resembles one.
    with pytest.raises(KeyError, match="language_model_extra"):
        list(MultimodalWeightStream([("language_model_extra.weight", 0)]))


def test_stream_of_lm_only_export_skips_nothing(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import MultimodalWeightStream  # noqa: PLC0415

    stream = MultimodalWeightStream([("language_model.lm_head.weight", 0)])
    assert list(stream) == [("lm_head.weight", 0)]
    assert sum(stream.skipped.values()) == 0
    assert stream.summary().endswith("skipped 0 vision tensor(s) (0 vision_tower.*, 0 connector.*)")


def test_loaded_names_are_reprefixed_for_vllms_strict_check(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import check_language_model_loaded  # noqa: PLC0415

    expected = ["model.embed_tokens.weight", "model.layers.0.mlp.experts.w13_weight"]
    result = check_language_model_loaded(set(expected), expected)
    assert result == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.mlp.experts.w13_weight",
    }
    # Loading more than named_parameters() lists (e.g. tied lm_head aliases) is not a failure.
    assert "language_model.lm_head.weight" in check_language_model_loaded(set(expected) | {"lm_head.weight"}, expected)


def test_missing_language_model_parameters_are_an_error(plugin_package) -> None:
    from olmoe3_vllm_plugin.multimodal_weights import check_language_model_loaded  # noqa: PLC0415

    with pytest.raises(RuntimeError, match=r"1 language-model parameter\(s\).*lm_head.weight"):
        check_language_model_loaded({"model.embed_tokens.weight"}, ["model.embed_tokens.weight", "lm_head.weight"])


# --- registration ------------------------------------------------------------------------------


def _fake_vllm(monkeypatch) -> list[tuple[str, object]]:
    registered: list[tuple[str, object]] = []
    registry = types.SimpleNamespace(register_model=lambda arch, cls: registered.append((arch, cls)))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(ModelRegistry=registry))
    return registered


def _fake_transformers_plugin(monkeypatch) -> list[int]:
    calls: list[int] = []
    module = types.ModuleType("olmoe3_transformers_plugin")
    module.register = lambda: calls.append(1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "olmoe3_transformers_plugin", module)
    return calls


def test_register_adds_the_vision_architecture_lazily_without_importing_it(
    plugin_package, monkeypatch, caplog
) -> None:
    registered = _fake_vllm(monkeypatch)
    calls = _fake_transformers_plugin(monkeypatch)
    # No transformers-side registration of `olmo3moe_vision` in this environment.
    monkeypatch.setattr(plugin_package, "_vision_config_registered", lambda: False)

    with caplog.at_level(logging.WARNING, logger="olmoe3_vllm_plugin"):
        plugin_package.register()

    assert calls == [1]
    assert registered == [
        ("Olmo3MoeForCausalLM", "olmoe3_vllm_plugin.model:Olmo3MoeForCausalLM"),
        (VISION_ARCHITECTURE, f"olmoe3_vllm_plugin.vision:{VISION_ARCHITECTURE}"),
    ]
    # Lazy targets: neither executor module is imported by registration, so the text model's
    # registration cannot be broken by the vision module (or vLLM's multimodal API) being absent.
    assert "olmoe3_vllm_plugin.model" not in sys.modules
    assert "olmoe3_vllm_plugin.vision" not in sys.modules
    assert "did not register model_type 'olmo3moe_vision'" in caplog.text


def test_register_is_quiet_when_the_transformers_plugin_knows_the_vision_config(
    plugin_package, monkeypatch, caplog
) -> None:
    registered = _fake_vllm(monkeypatch)
    _fake_transformers_plugin(monkeypatch)
    monkeypatch.setattr(plugin_package, "_vision_config_registered", lambda: True)

    with caplog.at_level(logging.WARNING, logger="olmoe3_vllm_plugin"):
        plugin_package.register()

    assert [arch for arch, _ in registered] == ["Olmo3MoeForCausalLM", VISION_ARCHITECTURE]
    assert "did not register model_type" not in caplog.text


def test_vision_config_lookup_asks_transformers_auto_mapping(plugin_package) -> None:
    pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # noqa: PLC0415

    assert plugin_package._vision_config_registered() == ("olmo3moe_vision" in CONFIG_MAPPING)


# --- executor contract (source level; the vLLM runtime has its own pinned environment) ---------


def test_vision_executor_serves_language_model_only() -> None:
    source = (PLUGIN / "src/olmoe3_vllm_plugin/vision.py").read_text()
    ast.parse(source)

    assert "from __future__ import annotations" not in source
    # SupportsMultiModal is what makes vLLM keep `language_model_only` (otherwise the flag is
    # dropped with multimodal_config); IsHybrid/HasInnerState route the KDA cache classmethods.
    assert "nn.Module, SupportsMultiModal, HasInnerState, IsHybrid, SupportsPP" in source
    assert "@MULTIMODAL_REGISTRY.register_processor(" in source
    assert 'return {"image": None}' in source
    assert 'mm_config.get_limit_per_prompt("image") == 0' in source
    assert "raise NotImplementedError(LANGUAGE_MODEL_ONLY_MESSAGE)" in source
    assert "with self._mark_language_model(vllm_config):" in source
    assert "init_vllm_registered_model(" in source
    assert "hf_config=config.text_config," in source
    assert 'prefix=maybe_prefix(prefix, "language_model")' in source
    assert "architectures=[TEXT_ARCHITECTURE]" in source
    assert 'TEXT_ARCHITECTURE = "Olmo3MoeForCausalLM"' in source
    # The hybrid classmethods are invoked on this class with the vision config; they must see
    # the text config.
    assert source.count("_text_vllm_config(vllm_config)") == 2
    assert "vllm_config.model_config.hf_config.text_config, architectures=[TEXT_ARCHITECTURE]" in source
    assert "Olmo3MoeForCausalLM.get_mamba_state_copy_func()" in source
    # Weight routing: LM tensors to the text executor, towers skipped and counted, strict return.
    assert "stream = MultimodalWeightStream(weights)" in source
    assert "self.language_model.load_weights(stream)" in source
    assert "check_language_model_loaded(loaded, expected)" in source
    assert "stream.summary()" in source
    # No tower modules are built, so vLLM's strict loader expects no vision parameters.
    assert "self.vision_tower" not in source and "self.connector" not in source
    assert "packed_modules_mapping = Olmo3MoeForCausalLM.packed_modules_mapping" in source


def test_vision_executor_is_registered_in_the_plugin_manifest_module() -> None:
    init = (PLUGIN / "src/olmoe3_vllm_plugin/__init__.py").read_text()
    assert f'VISION_ARCHITECTURE = "{VISION_ARCHITECTURE}"' in init
    assert f'"olmoe3_vllm_plugin.vision:{VISION_ARCHITECTURE}"' in init
    assert 'VISION_MODEL_TYPE = "olmo3moe_vision"' in init
    weights = (PLUGIN / "src/olmoe3_vllm_plugin/multimodal_weights.py").read_text()
    assert "import torch" not in weights and "import vllm" not in weights


# --- vLLM environment only ------------------------------------------------------------------


def test_vllm_registry_resolves_the_vision_executor_as_multimodal_hybrid() -> None:
    pytest.importorskip("vllm")
    # The transformers plugin registers OLMo-core's olmo3moe HF classes (allenai/OLMo-core#942 or later).
    pytest.importorskip("olmo_core.nn.moe.v2.hf")
    pytest.importorskip("olmoe3_transformers_plugin")
    import olmoe3_vllm_plugin  # noqa: PLC0415
    from vllm import ModelRegistry  # noqa: PLC0415

    olmoe3_vllm_plugin.register()
    # The lazy registration's own inspection: vLLM imports the executor module in a
    # subprocess (plugins loaded) and reads the class attributes the engine keys on.
    info = ModelRegistry.models[VISION_ARCHITECTURE].inspect_model_cls()
    assert info.supports_multimodal
    assert info.is_hybrid
    assert info.has_inner_state
    assert info.is_text_generation_model
    assert info.supports_pp
