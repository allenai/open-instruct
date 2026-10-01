"""Keep diagnostic comparisons aligned to actual response token positions."""

import math
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import emo_numerics
from scripts.miles.emo_trace_models import TraceMixin


def test_response_score_shift():
    logits = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [9.0, 0.0, 0.0]]])
    tokens = torch.tensor([[0, 0, 1, 2]])
    actual = emo_numerics.selected_scores(logits, tokens, prompt_length=2)
    expected = [-math.log1p(2 * math.exp(-2)), -math.log1p(2 * math.exp(-3))]
    assert actual == pytest.approx([float(x) for x in expected])


def test_comparison_reports_signed_bias_and_rejects_misalignment():
    result = emo_numerics.difference([-1.0, -1.0], [-0.5, -1.5])
    assert result == {"mean_abs": 0.5, "max_abs": 0.5, "signed_mean": 0.0}
    with pytest.raises(ValueError, match="aligned finite"):
        emo_numerics.difference([1.0, 2.0], [1.0])


def test_trace_checks_fused_weight_layout_and_exact_tokens(tmp_path):
    hf = emo_numerics.configuration_olmo3moe.Olmo3MoeConfig(
        vocab_size=16,
        hidden_size=16,
        attention_hidden_size=16,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=24,
        dense_layers_indices=[],
        layer_types=["full_attention"],
        use_head_qk_norm=True,
        use_rope=False,
    )
    model = emo_numerics.modeling_olmo3moe.Olmo3MoeForCausalLM(hf).eval()
    tokens = torch.tensor([[1, 3, 4]])
    outputs = {}
    handle = model.model.norm.register_forward_hook(lambda module, args, output: outputs.update(norm=output.detach()))
    with torch.no_grad():
        model(tokens, use_cache=False)
    handle.remove()
    state = model.state_dict()
    attn = "model.layers.0.self_attn."
    mlp = "model.layers.0.mlp."
    weights = {
        attn + "qkv_proj.weight": torch.cat([state[attn + x + ".weight"] for x in ("q_proj", "k_proj", "v_proj")]),
        mlp + "shared_expert.gate_up_proj.weight": torch.cat(
            [state[mlp + "shared_expert." + x + ".weight"] for x in ("gate_proj", "up_proj")]
        ),
        mlp + "experts.w13_weight": torch.stack(
            [torch.cat([state[f"{mlp}experts.{i}.{x}.weight"] for x in ("gate_proj", "up_proj")]) for i in range(4)]
        ),
        mlp + "experts.w2_weight": torch.stack([state[f"{mlp}experts.{i}.down_proj.weight"] for i in range(4)]),
        "model.norm.weight": state["model.norm.weight"],
    }
    trace = {"tokens": tokens.flatten(), "parameters": weights, "outputs": {"model.norm": outputs["norm"].squeeze(0)}}
    path = tmp_path / "trace.pt"
    torch.save(trace, path)
    result = emo_numerics.trace_comparisons(model, tokens, path)
    assert all(value["max_abs"] == 0 for value in result["parameters"].values())
    assert result["layers"]["model.norm"]["max_abs"] == 0
    weights[attn + "qkv_proj.weight"][0, 0] += 1
    torch.save(trace, path)
    result = emo_numerics.trace_comparisons(model, tokens, path)
    assert result["parameters"][attn + "qkv_proj.weight"]["max_abs"] == pytest.approx(1)
    with pytest.raises(ValueError, match="token IDs"):
        emo_numerics.trace_comparisons(model, tokens + 1, path)


def test_serving_trace_handles_direct_forward_and_preserves_prefill(monkeypatch, tmp_path):
    path = tmp_path / "trace.pt"
    monkeypatch.setenv("EMO_NUMERICS_TRACE", str(path))

    class Base(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model = torch.nn.Embedding(8, 4)

        def forward(self, input_ids, positions, forward_batch, input_embeds=None):
            return self.model(input_ids)

    class Model(TraceMixin, Base):
        pass

    model = Model(SimpleNamespace(hidden_size=4, num_hidden_layers=1, n_routed_experts=2))
    tokens = torch.tensor([1, 3, 4])
    expected = model.model(tokens).detach().clone()
    actual = model.forward(tokens, None, None)
    assert torch.equal(actual, expected)
    model.forward(tokens[:1], None, None)
    saved = torch.load(path, weights_only=True)
    assert torch.equal(saved["tokens"], tokens)
    assert torch.equal(saved["outputs"]["model"], expected)
    assert torch.equal(saved["parameters"]["model.weight"], model.model.weight)
