"""Release metadata and legacy-name regression checks, without checkpoint weights."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from jinja2 import Environment
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.olmo2 import Olmo2Config
from transformers.models.olmoe import OlmoeConfig

from open_instruct import hybrid_models, olmo_core_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.olmo_adapter import Olmo2ForSequenceClassification, OlmoeForSequenceClassification

FIXTURES = Path(__file__).parent / "test_data" / "hybrid_release"


@pytest.fixture(scope="module", autouse=True)
def register_models():
    hybrid_models.register_models()


@pytest.mark.parametrize("layout", ["primary", "science", "legacy", "explicit"])
def test_native_config_discovery(tmp_path, layout):
    if layout == "legacy":
        hf = tmp_path / "step38147-hf"
        native = tmp_path / "step38147"
    else:
        suffix = Path("v1/275m/midtrain")
        if layout == "science":
            suffix /= "from-cx025.000"
        hf = tmp_path / "hf" / suffix / "step0038147"
        native = tmp_path / "olmo-core" / suffix / "step0038147"
    hf.mkdir(parents=True)
    native.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "native_config.json", native / "config.json")
    source = native / "config.json" if layout == "explicit" else hf
    config = olmo_core_utils.get_transformer_config(str(source), 100352, "torch")
    assert config.d_model == 640
    assert config.n_layers == 10


def test_missing_release_native_config_fails_before_model_allocation(tmp_path):
    hf = tmp_path / "hf/v1/275m/midtrain/step0038147"
    hf.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="native_model_config"):
        olmo_core_utils.get_transformer_config(str(hf), 100352, "torch")


def test_release_model_and_weight_interchange_match_actual_checkpoint_header():
    hf_config = AutoConfig.from_pretrained(FIXTURES / "hf_config.json")
    native_config = olmo_core_utils.get_transformer_config(str(FIXTURES / "native_config.json"), 100352, "torch")
    with torch.device("meta"):
        hf = AutoModelForCausalLM.from_config(hf_config)
        native = native_config.build(init_device="meta")
    native_state = native.state_dict()
    converted = olmo_core_utils.convert_hybrid_small_state_to_hf(native_state, native_config.n_layers)
    expected_shapes = json.loads((FIXTURES / "tensor_shapes.json").read_text())
    assert {k: list(v.shape) for k, v in hf.state_dict().items()} == expected_shapes
    assert {k: list(v.shape) for k, v in converted.items()} == expected_shapes
    restored = olmo_core_utils._convert_hybrid_small_state_from_hf(converted, native_config.n_layers)
    assert set(restored) == set(native_state)
    assert all(restored[k] is native_state[k] for k in native_state)
    mapper = olmo_core_utils.make_hybrid_small_name_mapper(native_state)
    wrapped_mapper = olmo_core_utils.make_hybrid_small_name_mapper(
        k.replace(".attention.", ".attention._checkpoint_wrapped_module.") for k in native_state
    )
    for name, value in native_state.items():
        assert converted[mapper(name)] is value
        wrapped = name.replace(".attention.", ".attention._checkpoint_wrapped_module.")
        assert wrapped_mapper(wrapped) == mapper(name)


def test_legacy_model_name_still_loads_with_release_plugin():
    document = json.loads((FIXTURES / "hf_config.json").read_text())
    document.pop("model_type")
    config = AutoConfig.for_model("olmo_hybrid_small", **document)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    expected = json.loads((FIXTURES / "tensor_shapes.json").read_text())
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == expected


def test_base_prompt_format_has_no_thinking_or_chat_tokens():
    template = Environment().from_string(CHAT_TEMPLATES["gsm8k_base"])
    assert (
        template.render(
            messages=[{"role": "user", "content": "What is 2 + 2?"}], add_generation_prompt=True, eos_token="<eos>"
        )
        == "Question: What is 2 + 2?\nAnswer:"
    )
    assert (
        template.render(
            messages=[{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "4"}],
            add_generation_prompt=False,
            eos_token="<eos>",
        )
        == "Question: What is 2 + 2?\nAnswer: 4<eos>"
    )


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [(Olmo2Config, Olmo2ForSequenceClassification), (OlmoeConfig, OlmoeForSequenceClassification)],
)
def test_existing_classifier_default_survives_transformers_upgrade(config_class, model_class):
    config = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        eos_token_id=1,
    )
    config.id2label = None
    with torch.device("meta"):
        model = model_class(config)
    assert model.score.out_features == 2
