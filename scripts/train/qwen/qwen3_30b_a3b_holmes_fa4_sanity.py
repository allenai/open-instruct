import importlib.metadata
import json

import torch
from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, FusedAttentionV2
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.rope import RoPEConfig, RoPEType


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        raise RuntimeError(f"Expected a Blackwell GPU, found compute capability {props.major}.{props.minor}")

    torch.manual_seed(123)
    attention = FusedAttentionV2(
        d_model=2048,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        d_attn=4096,
        bias=False,
        rope=RoPEConfig(
            name=RoPEType.default,
            theta=1_000_000,
            full_precision=True,
        ),
        qk_norm=LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
            dtype=DType.bfloat16,
        ),
        backend=AttentionBackendName.flash_4,
        use_head_qk_norm=True,
        dtype=torch.bfloat16,
        init_device=str(device),
    ).train()

    x = torch.randn(1, 4096, 2048, dtype=torch.bfloat16, device=device, requires_grad=True)
    cu_doc_lens = torch.tensor([0, 1024, 2048, 4096], dtype=torch.int32, device=device)
    out = attention(x, cu_doc_lens=cu_doc_lens, max_doc_len=2048)
    out.float().square().mean().backward()
    torch.cuda.synchronize()

    if not torch.isfinite(out).all():
        raise RuntimeError("FA4 forward produced non-finite values")
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("FA4 backward produced missing or non-finite gradients")

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "flash_attn_4": importlib.metadata.version("flash-attn-4"),
                "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
                "device": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "output_mean": float(out.detach().float().mean()),
                "input_grad_norm": float(x.grad.float().norm()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
