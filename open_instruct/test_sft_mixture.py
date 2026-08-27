"""CPU tests for the generic SFT mixture layer (no weka, no GPU).

Molmo dataset construction is stubbed through SOURCE_REGISTRY so no real (weka-backed)
dataset is ever built; the weight math runs for real through
``mixture_weights.compute_flat_mixture_weights``.
"""

import math

import numpy as np
import pytest
from olmo_core.data.multimodal.mixtures import image_only_v9

from open_instruct import sft_mixture


class StubSource:
    """Minimal MixtureSource: deterministic length, records its spec."""

    def __init__(self, spec: sft_mixture.SourceSpec, length: int):
        self.spec = spec
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {"input_ids": np.arange(4, dtype=np.int64)}


# Deterministic fake lengths so sqrt-weight math is reproducible in tests.
_STUB_LENGTHS = {"tulu4": 400, "text_vqa": 100, "chart_qa_weighted": 2500, "open_instruct_text": 900}


@pytest.fixture
def stub_molmo_registry(monkeypatch):
    built: list[str] = []

    def factory(spec, tokenizer, *, seed, max_sequence_length):
        built.append(spec.name)
        return StubSource(spec, _STUB_LENGTHS.get(spec.name, 10_000))

    monkeypatch.setitem(sft_mixture.SOURCE_REGISTRY, sft_mixture.MOLMO_TYPE, factory)
    return built


def test_debug_mixture_builds_only_its_sources(stub_molmo_registry):
    config = sft_mixture.MixtureConfig(mixture="debug")
    datasets, weights, names = sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)
    assert set(names) == set(image_only_v9.DEBUG_MIXTURE_DATASETS)
    assert set(stub_molmo_registry) == set(image_only_v9.DEBUG_MIXTURE_DATASETS)
    assert len(datasets) == len(weights) == len(names)
    assert math.isclose(sum(weights), 1.0)


def test_debug_mixture_weight_math_matches_hand_computation(stub_molmo_registry):
    config = sft_mixture.MixtureConfig(mixture="debug")
    _, weights, names = sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)
    by_name = dict(zip(names, weights))
    # Group rates: nlp 0.166 (tulu4 alone), image_academic 0.418. Within image_academic,
    # text_vqa and chart_qa_weighted split by sqrt(len): sqrt(100)=10, sqrt(2500)=50.
    raw = {"tulu4": 0.166, "text_vqa": 0.418 * 10 / 60, "chart_qa_weighted": 0.418 * 50 / 60}
    norm = sum(raw.values())
    for name, expected in raw.items():
        assert math.isclose(by_name[name], expected / norm, rel_tol=1e-9), name


def test_nlp_source_open_instruct_swaps_tulu4(stub_molmo_registry, monkeypatch):
    adapter_specs: list[sft_mixture.SourceSpec] = []

    def adapter_factory(spec, tokenizer, *, seed, max_sequence_length):
        adapter_specs.append(spec)
        return StubSource(spec, _STUB_LENGTHS["open_instruct_text"])

    monkeypatch.setitem(sft_mixture.SOURCE_REGISTRY, sft_mixture.OPEN_INSTRUCT_SFT_TYPE, adapter_factory)
    config = sft_mixture.MixtureConfig(
        mixture="debug", nlp_source="open_instruct", nlp_rate=0.3, mixer_list=["allenai/Dolci-Instruct-SFT", "1.0"]
    )
    _, weights, names = sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)
    assert "tulu4" not in names
    assert sft_mixture.OPEN_INSTRUCT_TEXT_SOURCE_NAME in names
    assert adapter_specs[0].args["mixer_list"] == ["allenai/Dolci-Instruct-SFT", "1.0"]
    # nlp_rate=0.3 with image_academic 0.418 -> nlp weight = 0.3 / (0.3 + 0.418).
    by_name = dict(zip(names, weights))
    assert math.isclose(by_name[sft_mixture.OPEN_INSTRUCT_TEXT_SOURCE_NAME], 0.3 / (0.3 + 0.418), rel_tol=1e-9)


def test_generic_source_spec_extends_a_group(stub_molmo_registry, monkeypatch):
    monkeypatch.setitem(
        sft_mixture.SOURCE_REGISTRY,
        "custom_type",
        lambda spec, tokenizer, *, seed, max_sequence_length: StubSource(spec, 1600),
    )
    spec_json = '{"name": "my_source", "type": "custom_type", "group": "nlp", "rate": 0.5}'
    config = sft_mixture.MixtureConfig(mixture="debug", sources=[spec_json])
    _, weights, names = sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)
    assert "my_source" in names and "tulu4" not in names
    by_name = dict(zip(names, weights))
    assert math.isclose(by_name["my_source"], 0.5 / (0.5 + 0.418), rel_tol=1e-9)


def test_unknown_mixture_raises_listing_keys():
    with pytest.raises(ValueError, match="image-only-v9"):
        sft_mixture.MixtureConfig(mixture="bogus")


def test_unknown_source_type_raises(stub_molmo_registry):
    config = sft_mixture.MixtureConfig(mixture="debug", sources=['{"name": "x", "type": "nope", "group": "nlp"}'])
    with pytest.raises(ValueError, match="Unknown source type"):
        sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)


def test_source_spec_json_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown SourceSpec fields"):
        sft_mixture.SourceSpec.from_json('{"name": "x", "type": "molmo", "group": "nlp", "bogus": 1}')


def test_register_source_type_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        sft_mixture.register_source_type(sft_mixture.MOLMO_TYPE, lambda *a, **k: None)


def test_invalid_nlp_source_rejected():
    with pytest.raises(ValueError, match="nlp_source"):
        sft_mixture.MixtureConfig(nlp_source="bogus")


def test_full_mixture_resolves_all_groups(stub_molmo_registry):
    config = sft_mixture.MixtureConfig(mixture="image-only-v9")
    _, weights, names = sft_mixture.build_mixture(object(), config, max_sequence_length=64, seed=0)
    preset_names = {source.name for group in image_only_v9.IMAGE_ONLY_V9_SUBMIXTURES for source in group.datasets}
    assert set(names) == preset_names
    assert math.isclose(sum(weights), 1.0)
