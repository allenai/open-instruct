"""Preserve the pinned initial-loader quantization recipe during publication."""

import importlib
from functools import wraps

# DefaultModelLoader.load_weights_only scopes these settings while converting
# online FP4 weights, then restores the environment before serving. Distributed
# publication calls model.load_weights directly and otherwise misses that scope.
# Keep inference activation settings and all precision choices unchanged.
NVFP4_LOADER_ENV = {
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}


def install_nvfp4_publication_context(model_type=None):
    """Use initial-load FP4 math for every Olmo per-expert weight publication.

    Wrapping the model method covers both flattened buckets and ordinary NCCL
    tensors, retaining the native loader's w1/w3 staging across bucket boundaries.
    The native loader still chooses the kernel, scales and packing layout.
    """
    if model_type is None:
        model_type = importlib.import_module("olmo_sglang.models.olmo3_moe").Olmo3MoeForCausalLM
    original = model_type.load_weights
    if getattr(original, "_oi_nvfp4_publication_context", False):
        return

    @wraps(original)
    def load_weights(model, *values, **options):
        if not getattr(getattr(model, "quant_config", None), "is_nvfp4_online", False):
            return original(model, *values, **options)
        utils = importlib.import_module("sglang.srt.utils")
        with utils.temp_set_env(**NVFP4_LOADER_ENV):
            return original(model, *values, **options)

    load_weights._oi_nvfp4_publication_context = True
    model_type.load_weights = load_weights
