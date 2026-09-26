"""Process-local FP32 diagnostic accommodations for the pinned serving reference.

Loaded explicitly by a disposable sitecustomize in this experiment only. These
are not supported serving defaults or a production FP32 implementation.
"""

import collections
import functools
import json

import torch
from fla.ops.kda import chunk_intra
from olmo_sglang import core_compat
from olmo_sglang.kda import backend
from olmo_sglang.models import olmo3_moe
from torch.utils import _pytree
from triton import language as tl


def reject_low_precision(module, inputs, output):
    for value in _pytree.tree_leaves((inputs, output)):
        if isinstance(value, torch.Tensor) and value.dtype in (torch.bfloat16, torch.float16):
            raise ValueError(f"Low-precision tensor in FP32 reference: {type(module).__name__}, {value.dtype}")


def audit_model(model):
    counts = collections.Counter(str(p.dtype) for p in model.parameters())
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError(f"Non-FP32 model parameters: {counts}")
    for module in model.modules():
        module.register_forward_hook(reject_low_precision)
    print("FP32_MODEL_AUDIT", json.dumps(dict(counts)), flush=True)


def strict_arithmetic():
    """Also override FLA's explicit triangular-solve TF32 choice."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(chunk_intra, "SOLVE_TRIL_DOT_PRECISION"):
        chunk_intra.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


def install():
    """Allow FP32 in the eager reference and allocate its conv caches in FP32."""
    if getattr(core_compat, "_fp32_diagnostic_installed", False):
        return
    core_compat._fp32_diagnostic_installed = True
    strict_arithmetic()
    original_check = core_compat._is_bf16

    def allowed_reference_dtype(args, dtype):
        resolved = args.dtype if dtype is None else dtype
        return original_check(args, dtype) or resolved in ("float32", "fp32", torch.float32)

    core_compat._is_bf16 = allowed_reference_dtype
    backend._model_activation_dtype = lambda config: torch.float32
    original_init = olmo3_moe.Olmo3MoeForCausalLM.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        audit_model(self)

    olmo3_moe.Olmo3MoeForCausalLM.__init__ = init
