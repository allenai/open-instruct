"""Smoke tests for the multimodal APIs of the pinned ai2-olmo-core rev.

The multimodal SFT path (docs/design/multimodal_sft.md) imports these modules
lazily inside functions, so a pin moved to a rev that lacks them would
otherwise fail only at runtime on Beaker. These tests pin the contract at CI
time instead. They are import/attribute checks only -- no weka data, no GPU.
"""

import importlib

import pytest

pytest.importorskip("olmo_core", reason="ai2-olmo-core is not installed")


@pytest.mark.parametrize(
    ("module_name", "symbols"),
    [
        ("olmo_core.nn.vision.multimodal", ["MultimodalLM", "MultimodalLMConfig"]),
        (
            "olmo_core.nn.vision.molmo2_loader",
            [
                "molmo2_config_from_hf_config",
                "molmo2_hf_state_dict_to_multimodal_lm",
                "multimodal_lm_state_dict_to_hf",
                "retie_word_embeddings",
            ],
        ),
        ("olmo_core.nn.vision.molmo2_tokens", ["build_image_token_ids"]),
        (
            "olmo_core.train.train_module.transformer.multimodal_train_module",
            ["MultimodalTransformerTrainModule", "MultimodalTransformerTrainModuleConfig"],
        ),
        (
            "olmo_core.data.multimodal",
            [
                "MultimodalCollator",
                "MultimodalCollatorConfig",
                "MixtureDataLoader",
                "DatasetSource",
                "SubMixture",
                "compute_flat_mixture_weights",
                "Tulu4Dataset",
            ],
        ),
        ("olmo_core.data.multimodal.mixtures.image_only_v9", ["IMAGE_ONLY_V9_SUBMIXTURES", "VALIDATION_MIXTURES"]),
    ],
)
def test_multimodal_module_exports(module_name: str, symbols: list[str]) -> None:
    module = importlib.import_module(module_name)
    missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
    assert not missing, f"{module_name} is missing {missing}; was the ai2-olmo-core pin moved?"


def test_molmo2_presets_exist() -> None:
    multimodal = importlib.import_module("olmo_core.nn.vision.multimodal")
    for preset in ("molmo2_4B", "molmo2_8B"):
        assert callable(getattr(multimodal.MultimodalLMConfig, preset, None))
