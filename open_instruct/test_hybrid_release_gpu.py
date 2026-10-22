"""Actual release-weight parity, full-length update, and HF reload on one GPU."""

import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from open_instruct import hybrid_models, olmo_core_utils
from open_instruct.dataset_transformation import TokenizerConfig

CHECKPOINT = Path(
    os.environ.get(
        "OI_HYBRID_RELEASE_CHECKPOINT",
        "/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/275m/midtrain/step0038147",
    )
)


def assert_release_parity(native, hf, tokens):
    native_math, _ = olmo_core_utils.setup_model(
        olmo_core_utils.ModelConfig(model_name_or_path=str(CHECKPOINT), attn_implementation="torch"),
        init_device="meta",
    )
    native_math.to_empty(device="cuda")
    native_math.load_state_dict(native.state_dict(), strict=True)
    native_math.eval()
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected_fp32 = hf(tokens, use_cache=False).logits.float()
        actual_fp32 = native_math(tokens).float()
    print(f"FP32 math parity max_abs={(actual_fp32 - expected_fp32).abs().max().item():.8f}", flush=True)
    torch.testing.assert_close(actual_fp32, expected_fp32, rtol=5e-4, atol=1e-3)
    del native_math

    # FP16's additional mantissa bits separate model interchange errors from
    # accumulated BF16 rounding across eager/FlashAttention and fused kernels.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        expected_fp16 = hf(tokens, use_cache=False).logits.float()
        actual_fp16 = native(tokens).float()
    print(f"FP16 parity max_abs={(actual_fp16 - expected_fp16).abs().max().item():.6f}", flush=True)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = hf(tokens, use_cache=False).logits.float()
        actual = native(tokens).float()
    expected_logprobs = expected.log_softmax(-1)
    actual_logprobs = actual.log_softmax(-1)
    reference_logprobs = expected_fp32.log_softmax(-1)
    bf16_reference_kl = (reference_logprobs.exp() * (reference_logprobs - expected_logprobs)).sum(-1)
    fp16_reference_kl = (reference_logprobs.exp() * (reference_logprobs - expected_fp16.log_softmax(-1))).sum(-1)
    kl = (expected_logprobs.exp() * (expected_logprobs - actual_logprobs)).sum(-1)
    hf_precision_kl = (expected_logprobs.exp() * (expected_logprobs - expected_fp16.log_softmax(-1))).sum(-1)
    native_precision_kl = (actual_logprobs.exp() * (actual_logprobs - actual_fp16.log_softmax(-1))).sum(-1)
    print(
        f"BF16 parity max_abs={(actual - expected).abs().max().item():.6f} "
        f"mean_kl={kl.mean().item():.8f} max_kl={kl.max().item():.8f} "
        f"hf_precision_kl={hf_precision_kl.max().item():.8f} "
        f"native_precision_kl={native_precision_kl.max().item():.8f}",
        flush=True,
    )
    print(
        f"FP32 reference max_kl: BF16={bf16_reference_kl.max().item():.8f} FP16={fp16_reference_kl.max().item():.8f}",
        flush=True,
    )
    assert torch.isfinite(kl).all()
    assert kl.max().item() < 1e-3, "BF16 next-token distributions diverged"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="Requires the release checkpoint Weka mount")
def test_release_parity_update_and_export_reload(tmp_path):
    hybrid_models.register_models()
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(1)
    native, config = olmo_core_utils.setup_model(
        olmo_core_utils.ModelConfig(model_name_or_path=str(CHECKPOINT), attn_implementation="flash_2"),
        init_device="meta",
    )
    native.to_empty(device="cuda")
    state = native.state_dict()
    olmo_core_utils.load_hf_model_with_hybrid_support(str(CHECKPOINT), state, work_dir=str(tmp_path))
    native.load_state_dict(state, strict=True)
    del state
    tokenizer = TokenizerConfig(tokenizer_name_or_path=str(CHECKPOINT), chat_template_name="gsm8k_base").tokenizer
    tokens = torch.randint(0, config.vocab_size, (1, 2560), device="cuda")

    native.eval()
    hf = AutoModelForCausalLM.from_pretrained(str(CHECKPOINT), dtype=torch.float32, attn_implementation="eager")
    hf.to("cuda").eval()
    assert_release_parity(native, hf, tokens[:, :16])
    del hf

    native.train()
    optimizer = torch.optim.AdamW(native.parameters(), lr=1e-6)
    before = native.lm_head.norm.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = native(tokens)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
    assert torch.isfinite(loss)
    print(f"Full-length BF16 loss={loss.item():.6f}", flush=True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in native.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, native.lm_head.norm.weight)
    optimizer.zero_grad(set_to_none=True)
    del logits, loss

    native.eval()
    cpu_state = {name: value.detach().cpu() for name, value in native.state_dict().items()}
    output = tmp_path / "export"
    olmo_core_utils.save_state_dict_as_hf(cpu_state, str(output), str(CHECKPOINT), tokenizer)
    restored = AutoModelForCausalLM.from_pretrained(str(output), dtype=torch.float32, attn_implementation="eager")
    restored.to("cuda").eval()
    assert_release_parity(native, restored, tokens[:, :16])
    assert restored.config.model_type == "mainline_ladder"
    assert torch.cuda.max_memory_allocated() > 0
