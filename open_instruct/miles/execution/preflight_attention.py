"""Check Core's actual attention forward/backward before loading a large policy."""

import argparse
import json

import torch
from olmo_core.nn.attention import AttentionBackendName
from torch.nn import functional as F


def check(backend_name):
    torch.manual_seed(17)
    backend = AttentionBackendName(backend_name).build(head_dim=128, n_heads=16, n_kv_heads=8)
    values = [
        torch.randn(1, 64, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for heads in (16, 8, 8)
    ]
    reference_values = [value.detach().float().requires_grad_() for value in values]
    q, k, v = reference_values
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(2, dim=2).transpose(1, 2),
        v.repeat_interleave(2, dim=2).transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    output = backend(tuple(values))
    gradient = torch.randn_like(output)
    output.backward(gradient)
    reference.backward(gradient.float())
    torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.02)
    errors = []
    for value, ref in zip(values, reference_values, strict=True):
        assert value.grad is not None and ref.grad is not None
        torch.testing.assert_close(value.grad.float(), ref.grad, atol=0.03, rtol=0.05)
        errors.append((value.grad.float() - ref.grad).abs().max().item())
    report = dict(
        backend=backend_name,
        forward_max_abs=(output.float() - reference).abs().max().item(),
        gradient_max_abs=errors,
        passed=True,
    )
    print("CORE_ATTENTION_PREFLIGHT_PASSED", json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "flash_4"], default="flash_4")
    args = parser.parse_args()
    check(args.backend)


if __name__ == "__main__":
    main()
