"""Published Olmo 3 configuration fidelity and conservative scaling validation."""

import copy
import json
from pathlib import Path

import pytest
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from sglang.srt.utils.hf_transformers import config as serving_config
from transformers import AutoConfig, Olmo3Config

from open_instruct.miles import checkpoint, standard_models
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.errors import InputError


def published():
    return json.loads(Path(__file__).with_name("fixtures").joinpath("olmo3-think-dpo-config.json").read_text())


def test_published_config_maps_scaling_to_only_global_layers():
    hf = Olmo3Config(**published())
    config = standard_models.model_config_from_hf(hf, CoreConfig())
    assert config.n_layers == 32 and config.d_model == 4096
    assert config.block.sequence_mixer.rope.theta == 500000
    assert config.block.sequence_mixer.rope.scaling is None
    assert config.block.sequence_mixer.sliding_window.pattern == [4096, 4096, 4096, -1] * 8
    assert list(config.block_overrides) == list(range(3, 32, 4))
    for block in config.block_overrides.values():
        scaling = block.sequence_mixer.rope.scaling
        assert isinstance(scaling, YaRNRoPEScalingConfig)
        assert scaling.factor == 8 and scaling.old_context_len == 8192
        assert scaling.beta_fast == 32 and scaling.beta_slow == 1
        assert scaling.get_attention_rescale_factor() == published()["rope_scaling"]["attention_factor"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"factor": 0}, "factor"),
        ({"factor": True}, "factor"),
        ({"original_max_position_embeddings": 0}, "original_max_position_embeddings"),
        ({"beta_fast": 1, "beta_slow": 32}, "beta_fast"),
        ({"beta_fast": 1.5}, "integer"),
        ({"attention_factor": 1.0}, "attention_factor"),
        ({"mscale": 1.0}, "Unknown"),
        ({"rope_type": "dynamic"}, "explicit"),
    ],
)
def test_unsupported_yarn_semantics_fail_before_allocation(change, match):
    rope = published()["rope_scaling"] | change
    with pytest.raises(InputError, match=match):
        standard_models._rope_scaling("olmo3", rope)


def test_other_models_do_not_silently_adopt_olmo3_scaling():
    with pytest.raises(InputError, match="explicit"):
        standard_models._rope_scaling("qwen3", published()["rope_scaling"])


def test_layer_layout_is_validated():
    hf = Olmo3Config(**published())
    hf.layer_types = ["full_attention"]
    with pytest.raises(InputError, match="every layer"):
        standard_models.model_config_from_hf(hf, CoreConfig())


def test_descriptor_is_not_mutated_by_translation():
    hf = Olmo3Config(**published())
    before = copy.deepcopy(hf.to_dict())
    standard_models.model_config_from_hf(hf, CoreConfig())
    assert hf.to_dict() == before


def test_layer_overrides_survive_json_architecture_comparison():
    config = standard_models.model_config_from_hf(Olmo3Config(**published()), CoreConfig()).as_config_dict()
    saved = json.loads(json.dumps(config))
    assert checkpoint.comparable_model_config(config) == checkpoint.comparable_model_config(saved)
    saved["block_overrides"]["3"]["sequence_mixer"]["rope"]["scaling"]["factor"] = 4
    assert checkpoint.comparable_model_config(config) != checkpoint.comparable_model_config(saved)


@pytest.mark.parametrize("overrides", [{1: {}, "1": {}}, {"01": {}}, {-1: {}}, {True: {}}, {"garbage": {}}])
def test_layer_index_normalization_rejects_ambiguous_indices(overrides):
    with pytest.raises(ValueError, match="layer index"):
        checkpoint.comparable_model_config({"block_overrides": overrides})


def test_export_descriptor_loads_in_pinned_sglang_and_core(tmp_path):
    hf = Olmo3Config(**published())
    standard_models.save_hf_config(hf, tmp_path)
    document = json.loads((tmp_path / "config.json").read_text())
    assert "rope_parameters" not in document
    assert document["rope_scaling"] == published()["rope_scaling"]
    assert document["rope_theta"] == 500000
    serving = serving_config.get_config(str(tmp_path), trust_remote_code=False)
    assert serving.architectures == ["Olmo2ForCausalLM"]
    assert serving.rope_scaling["factor"] == 8
    assert serving.layer_types == hf.layer_types
    restored = AutoConfig.from_pretrained(tmp_path)
    original_core = standard_models.model_config_from_hf(hf, CoreConfig()).as_config_dict()
    restored_core = standard_models.model_config_from_hf(restored, CoreConfig()).as_config_dict()
    assert restored_core == original_core
