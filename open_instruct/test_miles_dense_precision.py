"""Dense Core optimizer precision regression checks in the pinned Core runtime."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Olmo2Config, Olmo3Config, Qwen3Config

pytest.importorskip("olmo_core.train.train_module.transformer.objective")
from open_instruct.miles import standard_models
from open_instruct.miles.config import CoreConfig


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    return torch.device(request.param)


@pytest.fixture(params=[Olmo2Config, Olmo3Config, Qwen3Config])
def dense_module(request, monkeypatch, device):
    torch.manual_seed(17)
    hf = request.param(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    hf._attn_implementation = "eager"
    reference = AutoModelForCausalLM.from_config(hf).to(device=device, dtype=torch.bfloat16).eval()
    options = CoreConfig(attention_backend="torch", activation_checkpointing=False)
    model = standard_models.model_config_from_hf(hf, options).build(init_device="meta")
    monkeypatch.setattr(standard_models.dist, "get_world_size", lambda: 1)
    module = standard_models.build_train_module(
        SimpleNamespace(olmo_core=options),
        common=dict(model=model, rank_microbatch_size=16, max_sequence_length=16, compile_model=False, device=device),
        optim=dict(lr=1e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        hf_config=hf,
        hf_state=reference.state_dict(),
    )
    return module, hf, reference


def test_dense_optimizer_accumulates_sub_bf16_updates(dense_module):
    module, _, _ = dense_module
    parameters = list(module.model.parameters())
    assert all(p.dtype == torch.float32 for p in parameters)
    parameter = parameters[0]
    with torch.no_grad():
        parameter.fill_(0.01)
    before = parameter.detach().clone()
    for _ in range(60):
        parameter.grad = torch.ones_like(parameter)
        module.optim.step()
        module.optim.zero_grad(set_to_none=True)
    assert torch.all(parameter.detach() < before)
    assert torch.any(parameter.detach().bfloat16() != before.bfloat16())
    state = module.optim.state[parameter]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_dense_bf16_forward_backward_and_publication(dense_module):
    module, hf, reference = dense_module
    ids = torch.tensor([[1, 2, 3, 4]], device=module.device)
    logits = module.model_forward(ids)
    assert logits.dtype == torch.bfloat16
    with torch.no_grad():
        expected = reference(ids).logits
    torch.testing.assert_close(logits.float(), expected.float(), atol=0.015, rtol=0.025)
    logits.float().square().mean().backward()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in module.model.parameters())
    exported = dict(standard_models.iter_export_state(module, hf))
    assert set(exported) == set(reference.state_dict())
    for name, value in exported.items():
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
