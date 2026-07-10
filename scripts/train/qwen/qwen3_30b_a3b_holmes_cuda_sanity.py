import json

import torch
import torch.nn.functional as F
from olmo_core.kernels.swiglu import swiglu_valid_prefix


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        raise RuntimeError(f"Expected a Blackwell GPU, found compute capability {props.major}.{props.minor}")
    if not torch.version.cuda or not torch.version.cuda.startswith("13."):
        raise RuntimeError(f"Expected a CUDA 13 PyTorch build, found {torch.version.cuda!r}")

    torch.manual_seed(123)
    x = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    num_elements = torch.tensor(24, device=device, dtype=torch.int64)
    actual = swiglu_valid_prefix(x, num_elements)
    expected = x[:24, :256] * F.silu(x[:24, 256:])
    torch.testing.assert_close(actual[:24], expected, rtol=0.02, atol=0.02)

    a = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    b = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    matmul_checksum = (a @ b).float().mean()
    torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "triton_version": __import__("triton").__version__,
                "device": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "swiglu_max_abs_diff": float((actual[:24] - expected).abs().max()),
                "matmul_checksum": float(matmul_checksum),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
