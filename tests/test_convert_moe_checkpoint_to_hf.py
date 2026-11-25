"""scripts/train/debug/convert_moe_checkpoint_to_hf.py writes KDA decay weights as float32, with the value
--kda-decay picks, and every other tensor as bfloat16. Read back from a small export on CPU."""

import importlib.util
import pathlib
import types
from copy import deepcopy

import pytest
import torch
from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, GateConfig, GateGranularity, KimiDeltaAttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.hf import convert_checkpoint_to_hf
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig, TransformerBlockType, TransformerType
from olmo_core.testing.utils import has_fla
from safetensors.torch import load_file

requires_fla = pytest.mark.skipif(not has_fla, reason="KDA needs flash-linear-attention with fla.ops.kda (Linux)")

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train" / "debug" / "convert_moe_checkpoint_to_hf.py"
)
spec = importlib.util.spec_from_file_location("convert_moe_checkpoint_to_hf", SCRIPT)
assert spec is not None and spec.loader is not None
convert_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_script)

WIDTH, EXPERTS, VOCAB = 128, 4, 64


def small_kda_config() -> OLMoDDPModelConfig:
    """Two KDA layers, one dense and one with routed experts, and the full-attention layer olmo-core's exporter
    requires beside them (after olmo-core's olmo3moe export fixture)."""
    norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False)
    sparse = OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=KimiDeltaAttentionConfig(
            n_heads=2, n_v_heads=2, head_dim=64, expand_v=2.0, allow_neg_eigval=True, use_experimental_kernels=False
        ),
        layer_norm=norm,
        shared_experts=SharedExpertsConfig(
            d_model=WIDTH, hidden_size=128, num_experts=1, bias=False, dtype=DType.float32
        ),
        routed_experts=RoutedExpertsConfig(
            d_model=WIDTH, hidden_size=128, num_experts=EXPERTS, bias=False, dtype=DType.float32
        ),
        routed_experts_router=MoERouterConfigV2(d_model=WIDTH, num_experts=EXPERTS, top_k=2),
        use_peri_norm=True,
        use_pre_norm=False,
    )
    dense = deepcopy(sparse)
    dense.routed_experts = dense.routed_experts_router = None
    full = deepcopy(sparse)
    full.sequence_mixer = AttentionConfig(
        n_heads=2,
        n_kv_heads=1,
        head_dim=64,
        bias=False,
        rope=None,
        qk_norm=deepcopy(norm),
        use_head_qk_norm=True,
        qk_norm_per_head_gains=True,
        scalable_softmax=True,
        backend=AttentionBackendName.torch,
        gate=GateConfig(granularity=GateGranularity.elementwise),
    )
    return OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=WIDTH,
        n_layers=3,
        vocab_size=VOCAB,
        block=sparse,
        block_overrides={0: dense, 1: deepcopy(sparse), 2: full},
        lm_head=LMHeadConfig(layer_norm=deepcopy(norm), bias=False),
        embedding_norm=deepcopy(norm),
        embed_scale=WIDTH**0.5,
        init_seed=42,
    )


def off_grid(tensor: torch.Tensor) -> torch.Tensor:
    """Values a third of the way between neighbouring bfloat16 values, so rounding to bfloat16 changes every one."""
    low = tensor.to(torch.bfloat16).to(torch.float32)
    high = torch.nextafter(low.to(torch.bfloat16), torch.tensor(torch.inf, dtype=torch.bfloat16)).to(torch.float32)
    return low + (high - low) / 3


@pytest.fixture(scope="module")
def checkpoint():
    """A small KDA model's config and float32 state dict, with every weight off the bfloat16 grid."""
    config = small_kda_config()
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=32)
    state = {key: off_grid(value.detach().to(torch.float32) + 0.25) for key, value in model.state_dict().items()}
    assert all(not torch.equal(v, v.to(torch.bfloat16).to(torch.float32)) for v in state.values())
    return config.as_config_dict(), state


def export(checkpoint, output_dir, monkeypatch, kda_decay):
    """Run olmo-core's converter as the script's main() does, with the tokenizer export stubbed (it needs the hub)."""
    config, state = checkpoint
    tokenizer = types.SimpleNamespace(
        bos_token_id=None, eos_token_id=VOCAB - 1, pad_token_id=VOCAB - 2, all_special_tokens=[], model_max_length=32
    )
    monkeypatch.setattr(convert_script.core_convert, "export_checkpoint_tokenizer", lambda *a, **k: tokenizer)
    with convert_script.kda_decay_export(kda_decay):
        convert_checkpoint_to_hf(
            original_checkpoint_path=None,
            output_path=output_dir,
            transformer_config_dict=deepcopy(config),
            tokenizer_config_dict={"vocab_size": VOCAB, "eos_token_id": VOCAB - 1, "pad_token_id": VOCAB - 2},
            model_state_dict={key: value.clone() for key, value in state.items()},
            dtype=DType.bfloat16,
            max_sequence_length=32,
            validate=False,
            device=torch.device("cpu"),
        )
    return load_file(str(pathlib.Path(output_dir) / "model.safetensors"))


def split(exported):
    decay = {k: v for k, v in exported.items() if k.endswith((".A_log", ".dt_bias"))}
    return decay, {k: v for k, v in exported.items() if k not in decay}


@requires_fla
@pytest.mark.parametrize("kda_decay", convert_script.KDA_DECAY_MODES)
def test_export_dtypes_and_decay_values(checkpoint, tmp_path, monkeypatch, kda_decay):
    _, state = checkpoint
    masters = {k: v for k, v in state.items() if k.endswith((".attention.A_log", ".attention.dt_bias"))}
    decay, others = split(export(checkpoint, tmp_path / "hf", monkeypatch, kda_decay))

    # Two KDA layers, two decay weights each, and nothing else in float32.
    assert len(decay) == len(masters) == 4
    assert {v.dtype for v in decay.values()} == {torch.float32}
    assert {v.dtype for v in others.values()} == {torch.bfloat16}
    assert any(".q_conv1d." in k for k in others) and any(".q_proj." in k for k in others)

    by_shape_and_value = sorted(decay.values(), key=lambda v: (v.numel(), v.sum().item()))
    expected = (
        masters.values() if kda_decay == "fp32" else [m.to(torch.bfloat16).to(torch.float32) for m in masters.values()]
    )
    for got, want in zip(by_shape_and_value, sorted(expected, key=lambda v: (v.numel(), v.sum().item())), strict=True):
        assert torch.equal(got.flatten(), want.flatten())


@requires_fla
def test_default_matches_the_all_bfloat16_export(checkpoint, tmp_path, monkeypatch):
    """bf16-grid changes only the container: cast to bfloat16, the export equals the one written before this option."""
    new = export(checkpoint, tmp_path / "new", monkeypatch, "bf16-grid")
    config, state = checkpoint
    tokenizer = types.SimpleNamespace(
        bos_token_id=None, eos_token_id=VOCAB - 1, pad_token_id=VOCAB - 2, all_special_tokens=[], model_max_length=32
    )
    monkeypatch.setattr(convert_script.core_convert, "export_checkpoint_tokenizer", lambda *a, **k: tokenizer)
    convert_checkpoint_to_hf(
        original_checkpoint_path=None,
        output_path=tmp_path / "old",
        transformer_config_dict=deepcopy(config),
        tokenizer_config_dict={"vocab_size": VOCAB, "eos_token_id": VOCAB - 1, "pad_token_id": VOCAB - 2},
        model_state_dict={key: value.clone() for key, value in state.items()},
        dtype=DType.bfloat16,
        max_sequence_length=32,
        validate=False,
        device=torch.device("cpu"),
    )
    old = load_file(str(tmp_path / "old" / "model.safetensors"))
    assert set(new) == set(old) and {v.dtype for v in old.values()} == {torch.bfloat16}
    for key, value in old.items():
        assert torch.equal(new[key].to(torch.bfloat16), value), key


def test_cast_refuses_a_state_dict_without_the_decay_weights():
    with pytest.raises(KeyError, match="A_log"):
        convert_script.cast_for_export({"w": torch.ones(2)}, {"blocks.0.attention.A_log"}, "fp32")
    with pytest.raises(ValueError, match="kda_decay"):
        convert_script.cast_for_export({"w": torch.ones(2)}, set(), "float32")


def test_fp32_needs_skip_validation(monkeypatch, capsys):
    """olmo-core's validation casts its reference model to one dtype, which would round the float32 decay weights."""
    monkeypatch.setattr("sys.argv", ["convert", "-i", "in", "-o", "out", "-c", "config.json", "--kda-decay", "fp32"])
    with pytest.raises(SystemExit):
        convert_script.main()
    assert "--skip-validation" in capsys.readouterr().err
