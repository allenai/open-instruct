import sys
import traceback

print("python", sys.version)

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())

print("---flash_attn---")
try:
    import flash_attn

    print("flash_attn", flash_attn.__version__, flash_attn.__file__)
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward  # noqa: F401

    print("flash_attn_interface ok")
except Exception:
    traceback.print_exc()

print("---ring_flash_attn---")
try:
    import ring_flash_attn

    print("ring_flash_attn", ring_flash_attn.__file__)
except Exception:
    traceback.print_exc()

print("---olmo_core has_ring_flash_attn---")
from olmo_core.nn.attention.flash_attn_api import has_ring_flash_attn

print("has_ring_flash_attn:", has_ring_flash_attn())
