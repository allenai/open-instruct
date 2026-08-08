"""Round-trip tests for the Qwen3.5 OLMo-core → HF exporter (qwen3_5_export.py).

The from-HF direction (`olmo_core.nn.hf.convert.convert_qwen3_5_state_from_hf`) is
OLMo-core's own trusted reference, so these tests prove the exporter is its exact
inverse: HF state → from_hf → to_hf must reproduce the original tensors. Runs on CPU
with a tiny randomly initialized `Qwen3_5ForCausalLM`; no GPU kernels are exercised
(that is the separate rung-2 logit-parity check).

Set QWEN3_5_REAL_CHECKPOINT=/path/to/hf/checkpoint to additionally round-trip a real
bf16 checkpoint (needs enough RAM to hold its state dict once).
"""

import json
import os

import pytest
import torch
import transformers
from olmo_core.nn.hf import convert as olmo_hf_convert
from olmo_core.nn.hf.convert import _split_qwen3_5_q_proj
from safetensors.torch import load_file
from transformers.models.qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from open_instruct import qwen3_5_export

_NORM_PATTERNS = qwen3_5_export._PLUS_ONE_NORM_PATTERNS


def _tiny_text_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
    )


def _hf_norm_keys(state: dict[str, torch.Tensor]) -> set[str]:
    """HF-side keys whose OLMo-core counterparts carry the +1.0 offset."""
    keys = set()
    for key in state:
        if (
            key.endswith(("input_layernorm.weight", "post_attention_layernorm.weight"))
            or key == "model.norm.weight"
            or key.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight"))
        ):
            keys.add(key)
    return keys


def _assert_roundtrip(config, hf_state: dict[str, torch.Tensor], atol: float, rtol: float = 0.0) -> None:
    olmo_state = olmo_hf_convert.convert_qwen3_5_state_from_hf(config, dict(hf_state))
    recovered = qwen3_5_export.convert_qwen3_5_state_to_hf(config, olmo_state)

    assert set(recovered) == set(hf_state), (
        f"key mismatch: missing={sorted(set(hf_state) - set(recovered))[:5]} "
        f"extra={sorted(set(recovered) - set(hf_state))[:5]}"
    )
    norm_keys = _hf_norm_keys(hf_state)
    for key, original in hf_state.items():
        result = recovered[key]
        assert result.shape == original.shape, f"{key}: shape {result.shape} != {original.shape}"
        if key in norm_keys:
            # These pass through +1.0 (load) then -1.0 (export); exactness is limited by the
            # dtype's ulp at |x + 1|, which scales with the weight's magnitude — hence rtol.
            torch.testing.assert_close(result, original, atol=atol, rtol=rtol, msg=key)
        else:
            assert torch.equal(result, original), f"{key}: tensors differ (should be untouched)"


def test_tiny_roundtrip_fp32():
    config = _tiny_text_config()
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config)
    hf_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _assert_roundtrip(config, hf_state, atol=1e-6)


def test_tiny_roundtrip_bf16():
    config = _tiny_text_config()
    torch.manual_seed(1)
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    hf_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # bf16 ulp around 1.0 is 2^-8; (x + 1) - 1 can move a near-zero norm entry by up to ~4e-3.
    _assert_roundtrip(config, hf_state, atol=4e-3)


def test_q_proj_interleave_is_exact_inverse():
    config = _tiny_text_config()
    n_heads, head_dim, hidden = config.num_attention_heads, config.head_dim, config.hidden_size
    torch.manual_seed(2)
    q_proj = torch.randn(n_heads * 2 * head_dim, hidden)
    w_q, w_g = _split_qwen3_5_q_proj(q_proj, n_heads, head_dim)
    merged = qwen3_5_export._merge_qwen3_5_q_proj(w_q, w_g, n_heads, head_dim)
    assert torch.equal(merged, q_proj)


def test_tied_embeddings_roundtrip_has_no_lm_head():
    config = _tiny_text_config()
    config.tie_word_embeddings = True
    torch.manual_seed(3)
    model = Qwen3_5ForCausalLM(config)
    hf_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    olmo_state = olmo_hf_convert.convert_qwen3_5_state_from_hf(config, dict(hf_state))
    recovered = qwen3_5_export.convert_qwen3_5_state_to_hf(config, olmo_state)
    assert "lm_head.weight" not in recovered
    assert set(recovered) == set(hf_state)


@pytest.mark.skipif(
    not os.environ.get("QWEN3_5_REAL_CHECKPOINT"), reason="set QWEN3_5_REAL_CHECKPOINT to run the full-size round-trip"
)
def test_real_checkpoint_roundtrip():
    path = os.environ["QWEN3_5_REAL_CHECKPOINT"]
    hf_config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_config = hf_config.get_text_config()

    hf_state: dict[str, torch.Tensor] = {}
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            hf_state.update(load_file(os.path.join(path, shard)))
    else:
        hf_state = load_file(os.path.join(path, "model.safetensors"))

    # Drop non-text weights (vision tower, MTP head) the same way the load path does; the
    # exporter targets the text-only serving layout.
    hf_state = {k: v for k, v in hf_state.items() if not any(s in k for s in ("visual.", "vision", "mtp."))}
    # Normalize any wrapper nesting to the in-memory layout the exporter emits.
    hf_state = {k.replace("model.language_model.", "model."): v for k, v in hf_state.items()}
    # Real trained norms drift to O(1) magnitudes (measured up to ~3.7 on the SFT anchor), so the
    # bf16 (x+1)-1 error is magnitude-proportional: one rounding at ulp(|x+1|) per direction.
    _assert_roundtrip(text_config, hf_state, atol=4e-3, rtol=8e-3)
