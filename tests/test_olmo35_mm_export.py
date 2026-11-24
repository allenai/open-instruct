# Ported from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/tests/test_multimodal_conversion.py
# (plugin path re-pointed at scripts/export/olmo35_mm/), plus open-instruct's own checks: the chat stop list in
# generation_config.json, and scripts/train/debug/convert_moe_checkpoint_to_hf.py's multimodal language-model path.
# Needs OLMo-core with nn.vision and nn.moe.v2.hf (allenai/OLMo-core#942); skipped otherwise.
"""CPU round trips of the multimodal (LM + vision + connector) OLMoDDP checkpoint export.

A tiny ``MultimodalLMConfig`` checkpoint is written in the layout ``MultimodalOLMoDDPTrainModule``
produces (``module.<name>.main`` masters, ``frozen_model.<name>`` frozen parameters) and
converted; its ``language_model.*`` part must be exactly the text path's export of the same LM.

The olmo-core forward of an exportable (routed-experts) OLMoDDP language model needs
Transformer Engine on a GPU, so the strict logit validation of the plugin converter is not run
here (``--skip-validation``, as the text path's own CPU tests). The exported wrapper's text
forward is instead compared with the text export's ``Olmo3MoeForCausalLM``.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from importlib import util
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN_SRC = REPO / "scripts" / "export" / "olmo35_mm" / "transformers_plugin" / "src"
CONVERT_MOE = REPO / "scripts" / "train" / "debug" / "convert_moe_checkpoint_to_hf.py"
CHAT_EOS = [100265, 100257]
_vision = pytest.importorskip("olmo_core.nn.vision")
if not hasattr(_vision, "MultimodalOLMoDDPModel"):
    pytest.skip("needs OLMo-core's multimodal OLMoDDP model (allenai/OLMo-core#942)", allow_module_level=True)
pytest.importorskip("olmo_core.nn.moe.v2.hf")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
tokenizers = pytest.importorskip("tokenizers")
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

# The skips above keep CI (whose OLMo-core has no multimodal code) green; these need OLMo-core#942.
from olmo_core.config import DType  # noqa: E402
from olmo_core.data.tokenizer import TokenizerConfig  # noqa: E402
from olmo_core.distributed.checkpoint import save_state_dict  # noqa: E402
from olmo_core.nn.attention import AttentionConfig  # noqa: E402
from olmo_core.nn.attention.backend import AttentionBackendName  # noqa: E402
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig  # noqa: E402
from olmo_core.nn.moe.emo import EmoRouterConfig  # noqa: E402
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig  # noqa: E402
from olmo_core.nn.moe.v2.router import MoERouterConfigV2  # noqa: E402
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig  # noqa: E402
from olmo_core.nn.rope import RoPEConfig  # noqa: E402
from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerBlockType, TransformerType  # noqa: E402
from olmo_core.nn.vision import (  # noqa: E402
    MultimodalLMConfig,
    MultimodalOLMoDDPModel,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmoe3_transformers_plugin import convert_checkpoint, register  # noqa: E402
from olmoe3_transformers_plugin.hf_vision.modeling_olmo3moe_vision import (  # noqa: E402
    Olmo3MoeVisionConnector,
    Olmo3MoeVisionForConditionalGeneration,
    Olmo3MoeVisionTower,
)
from olmoe3_transformers_plugin.multimodal import (  # noqa: E402
    has_multimodal_checkpoint_keys,
    is_multimodal_model_config,
    load_multimodal_state_dict,
)
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import decoders, models, pre_tokenizers  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM  # noqa: E402

VOCAB, EOS, PAD = 258, 256, 257  # byte-level BPE alphabet + <|endoftext|> + <|pad|>
D_MODEL = 32
IMAGE_PATCH_TOKEN = 1
TEXT_EXPORT_FILES = {
    "config.json",
    "configuration_olmo3moe.py",
    "generation_config.json",
    "model.safetensors",
    "modeling_olmo3moe.py",
    "moe_numerics.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer-export-audit.json",
}
VISION_CODE_FILES = {"configuration_olmo3moe_vision.py", "modeling_olmo3moe_vision.py"}


def _lm_config(routed_experts: bool = True):
    """A tiny OLMoDDP MoE LM the olmo3moe exporter accepts (RoPE attention, EMo router)."""
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
    block = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=AttentionConfig(
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            bias=False,
            rope=RoPEConfig(),
            qk_norm=deepcopy(norm),
            use_head_qk_norm=True,
            backend=AttentionBackendName.torch,
        ),
        layer_norm=norm,
        routed_experts=RoutedExpertsConfig(
            d_model=D_MODEL, hidden_size=32, num_experts=4, bias=False, dtype=DType.float32
        )
        if routed_experts
        else None,
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL,
            num_experts=4,
            top_k=2,
            emo=EmoRouterConfig(
                eos_token_id=EOS, min_document_expert_pool=2, max_document_expert_pool=4, eval_document_expert_pool=4
            ),
        )
        if routed_experts
        else None,
        shared_experts=None
        if routed_experts
        else SharedExpertsConfig(d_model=D_MODEL, hidden_size=32, num_experts=1, bias=False, dtype=DType.float32),
    )
    return OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=D_MODEL,
        n_layers=2,
        vocab_size=VOCAB,
        block=block,
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False),
        init_seed=42,
    )


def _vision_config(kind: str):
    common: dict[str, Any] = dict(
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=16,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=8,
        image_mlp_dim=24,
        image_norm_eps=1e-5,
    )
    if kind == "openai":  # CLIP-like: CLS token, pre-LN, no patch-embedding bias
        return VisionEncoderConfig(name=VisionEncoderType.openai, image_num_pos=5, **common)
    return VisionEncoderConfig(  # SigLIP-like: no CLS token, patch bias, no pre-LN
        name=VisionEncoderType.siglip,
        image_num_pos=4,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        **common,
    )


def _multimodal_config(kind: str = "siglip", routed_experts: bool = True):
    vision = _vision_config(kind)
    return MultimodalLMConfig(
        lm=_lm_config(routed_experts),
        vision=vision,
        connector=VisionConnectorConfig.from_vision_encoder(
            vision, output_dim=D_MODEL, num_input_layers=2, mlp_hidden_size=40
        ),
        image_patch_token_id=IMAGE_PATCH_TOKEN,
        vit_layers=(-1, -2),
    )


def _build_model(config, seed: int = 0):
    torch.manual_seed(seed)
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=16, device=torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # Non-uniform gains and biases, so a misplaced tensor cannot hide behind defaults.
            if name.endswith((".q_norm.weight", ".k_norm.weight", ".bias")):
                parameter.copy_(torch.linspace(0.5, 1.5, parameter.numel()).reshape(parameter.shape))
    return model.eval()


def _write_tokenizer(path: Path) -> None:
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = tokenizers.Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<|endoftext|>", "<|pad|>"])
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|pad|>"
    )
    assert (tokenizer.eos_token_id, tokenizer.pad_token_id, len(tokenizer)) == (EOS, PAD, VOCAB)
    tokenizer.save_pretrained(path)


def _write_checkpoint(step_dir: Path, model_config, entries: dict, tokenizer_dir: Path) -> dict:
    """Write an OLMoDDP-style step directory; ``entries`` maps checkpoint keys to tensors."""
    (step_dir / "model_and_optim").mkdir(parents=True)
    save_state_dict(str(step_dir / "model_and_optim"), entries)
    tokenizer = TokenizerConfig(vocab_size=VOCAB, eos_token_id=EOS, pad_token_id=PAD, identifier=str(tokenizer_dir))
    experiment = {
        "run_name": step_dir.parent.name,
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": tokenizer.as_config_dict()},
        "train_module": {"max_sequence_length": 16},
    }
    (step_dir / "config.json").write_text(json.dumps(experiment, indent=2))
    return experiment


def _multimodal_entries(model) -> dict:
    """The train module's layout: masters for trainable parameters, frozen entries otherwise.

    The vision tower and the LM head are frozen (as in the joint alignment stage); the rest of
    the LM and the connector train. Extra optimizer/skip-step entries are ignored by the export.
    """
    entries = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().clone()
        if name.startswith("vision.") or name == "lm.lm_head.w_out.weight":
            entries[f"frozen_model.{name}"] = tensor
        else:
            entries[f"module.{name}.main"] = tensor.view(-1)
            entries[f"module.{name}.exp_avg"] = torch.zeros_like(tensor.view(-1))
    entries["__moe_skip_step_losses"] = torch.zeros(4)
    return entries


def _text_entries(lm) -> dict:
    return {f"module.{name}.main": parameter.detach().clone().view(-1) for name, parameter in lm.named_parameters()}


def _convert(monkeypatch, step_dir: Path, output: Path, tokenizer_dir: Path, **flags) -> None:
    argv = [
        "convert_checkpoint",
        "-i",
        str(step_dir),
        "-o",
        str(output),
        "-t",
        str(tokenizer_dir),
        "-s",
        "16",
        "--skip-validation",
    ]
    for flag, value in flags.items():
        argv.extend([f"--{flag}", str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    convert_checkpoint.main()


def _read_tensors(path: Path) -> dict:
    return load_file(str(path / "model.safetensors"))


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    """Both exports of one model: the multimodal checkpoint and a text-only one of its LM."""
    root = tmp_path_factory.mktemp("exports")
    tokenizer_dir = root / "tokenizer"
    _write_tokenizer(tokenizer_dir)
    config = _multimodal_config("siglip")
    model = _build_model(config)
    mm_step = root / "multimodal" / "step10"
    experiment = _write_checkpoint(mm_step, config, _multimodal_entries(model), tokenizer_dir)
    text_step = root / "text" / "step10"
    _write_checkpoint(text_step, config.lm, _text_entries(model.lm), tokenizer_dir)
    # Distinct basenames: transformers keys the remote-code modules of a local export by its
    # directory name, and reloading a same-named module orphans an already built model.
    multimodal_hf = root / "multimodal" / "hf-multimodal"
    text_hf = root / "text" / "hf-text"
    monkeypatch = pytest.MonkeyPatch()
    try:
        _convert(monkeypatch, mm_step, multimodal_hf, tokenizer_dir)
        _convert(monkeypatch, text_step, text_hf, tokenizer_dir)
    finally:
        monkeypatch.undo()
    return {"model": model, "experiment": experiment, "multimodal": multimodal_hf, "text": text_hf}


def test_detects_multimodal_configs_and_checkpoints():
    assert is_multimodal_model_config(_multimodal_config().as_config_dict())
    assert not is_multimodal_model_config(_lm_config().as_config_dict())
    assert has_multimodal_checkpoint_keys({"module.lm.embeddings.weight.main"})
    assert has_multimodal_checkpoint_keys({"frozen_model.vision.patch_embedding.weight"})
    assert not has_multimodal_checkpoint_keys({"module.embeddings.weight.main", "__skip"})


def test_language_model_tensors_are_the_text_export(exports):
    multimodal = _read_tensors(exports["multimodal"])
    text = _read_tensors(exports["text"])
    language_model = {
        name[len("language_model.") :]: tensor
        for name, tensor in multimodal.items()
        if name.startswith("language_model.")
    }
    assert set(language_model) == set(text)
    for name, tensor in text.items():
        assert tensor.dtype == torch.bfloat16
        assert language_model[name].dtype == tensor.dtype
        assert torch.equal(language_model[name], tensor), name


def test_vision_and_connector_tensors_keep_olmo_core_names(exports):
    model = exports["model"]
    multimodal = _read_tensors(exports["multimodal"])
    for prefix, module in (("vision_tower.", model.vision), ("connector.", model.connector)):
        exported = {name: t for name, t in multimodal.items() if name.startswith(prefix)}
        reference = module.state_dict()
        assert set(exported) == {f"{prefix}{name}" for name in reference}
        for name, tensor in reference.items():
            assert exported[f"{prefix}{name}"].dtype == torch.bfloat16
            assert torch.equal(exported[f"{prefix}{name}"], tensor.to(torch.bfloat16)), name
    assert set(multimodal) == {
        name for name in multimodal if name.startswith(("language_model.", "vision_tower.", "connector."))
    }


def test_config_combines_the_text_export_with_the_olmo_core_configs(exports):
    config = json.loads((exports["multimodal"] / "config.json").read_text())
    text_config = json.loads((exports["text"] / "config.json").read_text())
    model_config = exports["experiment"]["model"]
    assert config["architectures"] == ["Olmo3MoeVisionForConditionalGeneration"]
    assert config["model_type"] == "olmo3moe_vision"
    assert config["text_config"] == text_config
    assert "auto_map" in config["text_config"]
    assert config["vision_config"] == model_config["vision"]
    assert config["connector_config"] == model_config["connector"]
    assert config["image_patch_token_id"] == IMAGE_PATCH_TOKEN
    assert config["vit_layers"] == [-1, -2]
    assert config["olmo_core_config"] == exports["experiment"]
    assert config["auto_map"] == {
        "AutoConfig": "configuration_olmo3moe_vision.Olmo3MoeVisionConfig",
        "AutoModel": "modeling_olmo3moe_vision.Olmo3MoeVisionForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_olmo3moe_vision.Olmo3MoeVisionForConditionalGeneration",
    }
    for name in ("eos_token_id", "pad_token_id", "bos_token_id", "dtype"):
        assert config[name] == text_config[name]


def test_export_files_are_the_text_export_plus_the_wrapper_code(exports):
    text_files = {path.name for path in exports["text"].iterdir()}
    multimodal_files = {path.name for path in exports["multimodal"].iterdir()}
    assert text_files == TEXT_EXPORT_FILES
    assert multimodal_files == TEXT_EXPORT_FILES | VISION_CODE_FILES
    for name in TEXT_EXPORT_FILES - {"config.json", "model.safetensors"}:
        assert (exports["multimodal"] / name).read_bytes() == (exports["text"] / name).read_bytes()
    package_dir = PLUGIN_SRC / "olmoe3_transformers_plugin" / "hf_vision"
    for name in VISION_CODE_FILES:
        assert (exports["multimodal"] / name).read_bytes() == (package_dir / name).read_bytes()


def test_wrapper_loads_with_remote_code_and_runs_the_text_model(exports):
    torch.manual_seed(1)
    input_ids = torch.randint(2, VOCAB, (2, 9))
    text_model = AutoModelForCausalLM.from_pretrained(exports["text"], trust_remote_code=True)
    with torch.no_grad():
        expected = text_model(input_ids=input_ids).logits

    config = AutoConfig.from_pretrained(exports["multimodal"], trust_remote_code=True)
    assert type(config).__name__ == "Olmo3MoeVisionConfig"
    assert type(config.text_config).__name__ == "Olmo3MoeConfig"
    assert config.get_text_config() is config.text_config
    wrapper, info = AutoModel.from_pretrained(exports["multimodal"], trust_remote_code=True, output_loading_info=True)
    assert type(wrapper).__name__ == "Olmo3MoeVisionForConditionalGeneration"
    assert type(wrapper.language_model).__name__ == "Olmo3MoeForCausalLM"
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert not info.get("mismatched_keys")
    assert wrapper.language_model.lm_head.weight.dtype == torch.bfloat16
    with torch.no_grad():
        logits = wrapper(input_ids=input_ids).logits
    assert logits.shape == (2, 9, VOCAB)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="[Vv]ision"):
        wrapper(input_ids=input_ids, images=torch.zeros(2, 1, 4, 14 * 14 * 3))


def test_wrapper_loads_through_the_registered_auto_classes(exports):
    register()
    wrapper = AutoModelForCausalLM.from_pretrained(exports["multimodal"])
    assert isinstance(wrapper, Olmo3MoeVisionForConditionalGeneration)
    text_model = AutoModelForCausalLM.from_pretrained(exports["text"])
    input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        torch.testing.assert_close(
            wrapper(input_ids=input_ids).logits, text_model(input_ids=input_ids).logits, rtol=0, atol=0
        )


def test_multimodal_state_dict_loads_every_parameter_kind(exports, tmp_path):
    """Masters, frozen entries and buffers resolve exactly as the train module stores them."""
    model = exports["model"]
    step_dir = exports["multimodal"].parent / "step10"
    experiment = json.loads((step_dir / "config.json").read_text())
    state = load_multimodal_state_dict(str(step_dir), experiment["model"])
    reference = model.state_dict()
    assert set(state) == set(reference)
    for name, tensor in reference.items():
        assert torch.equal(state[name], tensor), name


@pytest.mark.parametrize("kind", ["openai", "siglip"])
def test_wrapper_vision_modules_mirror_olmo_core_shapes(kind):
    config = _multimodal_config(kind)
    model = config.build(init_device="meta")
    tower = Olmo3MoeVisionTower(config.vision.as_config_dict())
    connector = Olmo3MoeVisionConnector(config.connector.as_config_dict())
    for mirror, reference in ((tower, model.vision), (connector, model.connector)):
        expected = {name: tuple(t.shape) for name, t in reference.state_dict().items()}
        actual = {name: tuple(t.shape) for name, t in mirror.state_dict().items()}
        assert actual == expected


def test_text_only_checkpoint_takes_the_text_path(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer(tokenizer_dir)
    config = _lm_config()
    lm = _build_model(config)
    step_dir = tmp_path / "text" / "step10"
    _write_checkpoint(step_dir, config, _text_entries(lm), tokenizer_dir)
    monkeypatch.setattr(
        convert_checkpoint,
        "convert_multimodal_checkpoint",
        lambda *args, **kwargs: pytest.fail("text checkpoint routed to the multimodal path"),
    )
    _convert(monkeypatch, step_dir, tmp_path / "hf", tokenizer_dir)
    config_json = json.loads((tmp_path / "hf" / "config.json").read_text())
    assert config_json["model_type"] == "olmo3moe"
    assert set(_read_tensors(tmp_path / "hf")) >= {"lm_head.weight", "model.embed_tokens.weight"}


def test_multimodal_text_only_forward_is_the_language_model_forward():
    """Validating the exported LM alone covers the multimodal model's text forward."""
    # A dense LM (no routed experts): the only OLMoDDP forward that runs without TE/GPU.
    model = _build_model(_multimodal_config("siglip", routed_experts=False))
    assert isinstance(model, MultimodalOLMoDDPModel)
    torch.manual_seed(2)
    input_ids = torch.randint(2, VOCAB, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids), model.lm(input_ids))


# --- open-instruct additions -----------------------------------------------------------------------


def test_generation_config_stops_at_im_end(exports):
    """Both exports stop at <|im_end|> as well as <|endoftext|> (H271: vLLM ran past every tool call)."""
    for kind in ("multimodal", "text"):
        generation_config = json.loads((exports[kind] / "generation_config.json").read_text())
        assert generation_config["eos_token_id"] == CHAT_EOS, kind


def _load_convert_moe():
    spec = util.spec_from_file_location("convert_moe_checkpoint_to_hf", CONVERT_MOE)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_convert_moe(monkeypatch, *argv) -> None:
    module = _load_convert_moe()
    monkeypatch.setattr(
        sys, "argv", ["convert_moe_checkpoint_to_hf.py", *argv, "--skip-validation", "--device", "cpu"]
    )
    module.main()


def _assert_is_the_text_export(path: Path, exports) -> None:
    exported = _read_tensors(path)
    reference = _read_tensors(exports["text"])
    assert set(exported) == set(reference)
    for name, tensor in reference.items():
        assert torch.equal(exported[name], tensor), name
    assert json.loads((path / "generation_config.json").read_text())["eos_token_id"] == CHAT_EOS
    config = json.loads((path / "config.json").read_text())
    assert config["model_type"] == "olmo3moe"
    assert config["max_position_embeddings"] == 16  # the run's train_module.max_sequence_length


@pytest.mark.parametrize("layout", ["olmoddp", "olmo_core_model"])
def test_convert_moe_exports_the_language_model_of_a_multimodal_checkpoint(exports, monkeypatch, tmp_path, layout):
    """A DCP step dir: OLMoDDP masters (GPU run) or ``model.*`` entries (a TPU run converted to OLMo-core)."""
    step_dir = exports["multimodal"].parent / "step10"
    if layout == "olmo_core_model":
        model = exports["model"]
        step_dir = tmp_path / "tpu" / "step10"
        (step_dir / "model_and_optim").mkdir(parents=True)
        save_state_dict(
            str(step_dir / "model_and_optim"),
            {f"model.{k}": v.detach().clone() for k, v in model.state_dict().items()},
        )
        (step_dir / "config.json").write_text(json.dumps(exports["experiment"]))
    out = tmp_path / f"hf-text-{layout}"
    tokenizer_dir = exports["multimodal"].parent.parent / "tokenizer"
    _run_convert_moe(
        monkeypatch, "-i", str(step_dir), "-c", str(step_dir / "config.json"), "-o", str(out), "-t", str(tokenizer_dir)
    )
    _assert_is_the_text_export(out, exports)


def test_convert_moe_exports_the_language_model_of_a_consolidated_export(exports, monkeypatch, tmp_path):
    """OLMo-core's export_multimodal_olmoddp.py layout: model.safetensors (bf16) + olmo_core_config.json."""
    export_dir = tmp_path / "olmo-core-export"
    export_dir.mkdir()
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in exports["model"].state_dict().items()}
    save_file(state, str(export_dir / "model.safetensors"))
    (export_dir / "olmo_core_config.json").write_text(json.dumps(exports["experiment"]))
    out = tmp_path / "hf-text-consolidated"
    tokenizer_dir = exports["multimodal"].parent.parent / "tokenizer"
    _run_convert_moe(monkeypatch, "--consolidated-export", str(export_dir), "-o", str(out), "-t", str(tokenizer_dir))
    _assert_is_the_text_export(out, exports)


def test_convert_moe_rejects_a_text_only_consolidated_export(exports, monkeypatch, tmp_path):
    export_dir = tmp_path / "text-export"
    export_dir.mkdir()
    save_file({"embeddings.weight": torch.zeros(2, 2)}, str(export_dir / "model.safetensors"))
    (export_dir / "olmo_core_config.json").write_text(json.dumps({"model": {}}))
    with pytest.raises(RuntimeError, match="not a multimodal export"):
        _run_convert_moe(monkeypatch, "--consolidated-export", str(export_dir), "-o", str(tmp_path / "out"))


def test_tokenizer_config_is_found_in_multi_source_runs():
    module = _load_convert_moe()
    tokenizer = {"vocab_size": 7, "eos_token_id": 1}
    assert module.resolve_tokenizer_config({"dataset": {"tokenizer": tokenizer}}) == tokenizer
    multi = {"dataset": [{"tokenizer": tokenizer}, {"tokenizer": dict(tokenizer)}]}
    assert module.resolve_tokenizer_config(multi) == tokenizer
    with pytest.raises(RuntimeError, match="disagreeing"):
        module.resolve_tokenizer_config({"dataset": [{"tokenizer": tokenizer}, {"tokenizer": {"vocab_size": 8}}]})


def test_inference_kernels_replace_training_only_choices():
    module = _load_convert_moe()
    config: dict[str, Any] = {
        "block": {"sequence_mixer": {"use_experimental_kernels": True}},
        "block_overrides": {
            "7": {"sequence_mixer": {"_CLASS_": "olmo_core.nn.attention.AttentionConfig", "backend": "flash_4"}}
        },
    }
    module.use_inference_kernels(config)
    assert config["block"]["sequence_mixer"]["use_experimental_kernels"] is False
    assert config["block_overrides"]["7"]["sequence_mixer"]["backend"] == "torch"
