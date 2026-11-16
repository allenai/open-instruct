"""Online FP8 publication compatibility for the pinned SGLang/Olmo runtime."""

import importlib
from functools import wraps

import torch


def _online(method):
    return (
        method.quant_config.get_name() == "fp8"
        and not method.quant_config.is_checkpoint_fp8_serialized
        and not method.block_quant
        and not getattr(method, "is_fp4_expert", False)
    )


def _install_method(method_type, supported):
    """Reuse native allocation/quantization, retaining serving storage at finalize.

    SGLang b84bec's FP8 methods replace Parameters during initial conversion but
    inherit a no-op restore hook. Replaying BF16 updates into those FP8 tensors
    loses precision (and linear weights have also been transposed). Keep only
    allocation arguments between updates, not a persistent BF16 model copy.
    """
    create = method_type.create_weights
    if getattr(create, "_oi_fp8_publication", False):
        return
    restore = method_type.restore_weights_before_loading
    process = method_type.process_weights_after_loading

    @wraps(create)
    def create_weights(method, layer, *args, **kwargs):
        result = create(method, layer, *args, **kwargs)
        if _online(method):
            layer._oi_fp8_allocation = (args, kwargs.copy())
        return result

    @wraps(restore)
    def restore_weights(method, layer):
        if not _online(method):
            return restore(method, layer)
        if not supported(method):
            raise RuntimeError("Online FP8 refresh requires CUDA, Triton MoE and non-Marlin linear kernels")
        if hasattr(layer, "_oi_fp8_serving_parameters"):
            raise RuntimeError("FP8 refresh is already open")
        serving = dict(layer.named_parameters(recurse=False))
        args, kwargs = layer._oi_fp8_allocation
        # The original allocator restores BF16 shape, dtype and weight_loader
        # attributes, including separate w1/w3 slices and linear sharding.
        with torch.device(next(iter(serving.values())).device):
            create(method, layer, *args, **kwargs)
        layer._oi_fp8_serving_parameters = serving

    @wraps(process)
    def process_weights(method, layer):
        result = process(method, layer)
        serving = getattr(layer, "_oi_fp8_serving_parameters", None)
        if serving is None:
            return result
        updated = dict(layer.named_parameters(recurse=False))
        # Validate the whole layer before copying. Unsupported layout changes
        # must fail the publication, never silently invalidate captured graphs.
        if serving.keys() != updated.keys():
            raise RuntimeError("FP8 refresh changed the serving parameter inventory")
        for name, previous in serving.items():
            current = updated[name]
            if (previous.shape, previous.dtype, previous.device, previous.stride()) != (
                current.shape,
                current.dtype,
                current.device,
                current.stride(),
            ):
                raise RuntimeError(f"FP8 refresh changed serving layout for {name}")
        with torch.no_grad():
            for name, previous in serving.items():
                previous.copy_(updated[name])
                setattr(layer, name, previous)
        del layer._oi_fp8_serving_parameters
        return result

    create_weights._oi_fp8_publication = True
    method_type.create_weights = create_weights
    method_type.restore_weights_before_loading = restore_weights
    method_type.process_weights_after_loading = process_weights


def _install_loader(model_type):
    original = model_type.load_weights
    if getattr(original, "_oi_fp8_publication", False):
        return

    @wraps(original)
    def load_weights(model, *args, **kwargs):
        config = getattr(model, "quant_config", None)
        if config is not None and config.get_name() == "fp8":
            # Olmo 069dc0 caches Parameter objects, not just name resolution.
            # Refresh it after native allocation has replaced the load targets.
            model._weight_targets = None
        return original(model, *args, **kwargs)

    load_weights._oi_fp8_publication = True
    model_type.load_weights = load_weights


def install_fp8_publication():
    """Install before model construction; leave NVFP4 and BF16 unchanged."""
    fp8 = importlib.import_module("sglang.srt.layers.quantization.fp8")
    _install_method(fp8.Fp8MoEMethod, lambda method: fp8._is_cuda and fp8.get_moe_runner_backend().is_triton())
    _install_method(fp8.Fp8LinearMethod, lambda method: fp8._is_cuda and not method.use_marlin)
    _install_loader(importlib.import_module("olmo_sglang.models.olmo3_moe").Olmo3MoeForCausalLM)
