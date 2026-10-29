# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import torch
import transformers

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

SUPPORTED_ROPE_TYPES = {"default", "yarn", "linear", "llama3"}
VALID_ROTARY_LAYER_TYPES = {"full_attention", "sliding_attention"}


def parse_rotary_buffer_name(
    mod: torch.nn.Module, bname: str, config: transformers.PretrainedConfig | None = None
) -> tuple[bool, str | None]:
    """
    Validates and parses a buffer name to ensure it belongs to an actual RoPE module.

    Rejects arbitrary '*_inv_freq' names. Only recognizes:
    - 'inv_freq'
    - 'original_inv_freq'
    - '<layer_type>_inv_freq' and '<layer_type>_original_inv_freq' where layer_type
      is explicitly defined in a nested config.rope_parameters dict or is a recognized
      OLMo 3 layer type. Flat rope_parameters keys (e.g. rope_theta, factor) are rejected.

    Returns:
        (is_rotary, layer_type)
    """
    # 1. Module sanity check: must be a rotary/rope module or have rotary attributes
    mod_name = mod.__class__.__name__.lower()
    is_rotary_module = (
        "rotary" in mod_name
        or "rope" in mod_name
        or hasattr(mod, "rope_type")
        or hasattr(mod, "compute_default_rope_parameters")
    )
    if not is_rotary_module:
        return False, None

    # 2. Standard single-layout buffers
    if bname in ("inv_freq", "original_inv_freq"):
        return True, None

    # 3. Typed per-layer buffers: determine valid layer types from nested config or known types
    valid_layer_types = set(VALID_ROTARY_LAYER_TYPES)
    cfg = config if config is not None else getattr(mod, "config", None)
    if cfg is not None:
        rope_params = getattr(cfg, "rope_parameters", None)
        # Only treat keys as layer types if rope_parameters is genuinely nested (values are dicts)
        if isinstance(rope_params, dict):
            nested_types = {k for k, v in rope_params.items() if isinstance(v, dict)}
            valid_layer_types.update(nested_types)

    for lt in valid_layer_types:
        if bname in (f"{lt}_inv_freq", f"{lt}_original_inv_freq"):
            return True, lt

    return False, None


def _call_with_supported_kwargs(fn, *args, **kwargs):
    """Invokes fn passing only kwargs that exist in its signature or if **kwargs is accepted."""
    try:
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = kwargs if has_varkw else {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **filtered_kwargs)
    except (ValueError, TypeError):
        return fn(*args, **kwargs)


def recompute_rotary_inv_freq(
    mod: torch.nn.Module,
    config: transformers.PretrainedConfig | None = None,
    device: torch.device | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float | None]:
    """
    Recomputes canonical RoPE inverse frequency buffers in torch.float32 from authoritative config.

    Explicitly supports:
    - 'default': canonical 1.0 / (rope_theta ** (arange / dim))
    - 'yarn': YaRN attention scaling and frequency interpolation
    - 'linear': linear interpolation scaling
    - 'llama3': LLaMA 3 frequency scaling

    Preserves nested configuration for per-layer RoPE and passes layer_type to native computation
    functions. Supports partial_rotary_factor for effective dimension calculation.

    Raises NotImplementedError for unsupported or unverified scaling types (e.g. dynamic, longrope).
    Never silently replaces unknown scaling with default frequencies.
    """
    if config is None:
        config = getattr(mod, "config", None)
    if config is None:
        raise ValueError("Cannot recompute RoPE frequencies without an authoritative config.")

    cfg = config
    rope_params = getattr(cfg, "rope_parameters", None)
    is_nested_params = isinstance(rope_params, dict) and any(isinstance(v, dict) for v in rope_params.values())

    # Look up parameters for this layer_type if nested
    layer_rope_dict: dict = {}
    if is_nested_params and layer_type and layer_type in rope_params:
        layer_rope_dict = rope_params[layer_type]
    elif isinstance(rope_params, dict) and not is_nested_params:
        layer_rope_dict = rope_params

    # Determine rope_type
    rope_type = getattr(mod, "rope_type", None)
    if rope_type is None or layer_type:
        if layer_rope_dict:
            rope_type = layer_rope_dict.get("rope_type", "default")
        else:
            rope_scaling = getattr(cfg, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            else:
                rope_type = "default"

    # Explicitly reject unsupported variants - never silently default!
    if rope_type not in SUPPORTED_ROPE_TYPES:
        raise NotImplementedError(
            f"RoPE scaling type '{rope_type}' is not supported for fp32 restoration. "
            f"Verified supported types: {sorted(SUPPORTED_ROPE_TYPES)}."
        )

    # Calculate effective rotary dimension respecting partial_rotary_factor
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    partial_rotary_factor = 1.0
    if layer_rope_dict and "partial_rotary_factor" in layer_rope_dict:
        partial_rotary_factor = float(layer_rope_dict["partial_rotary_factor"])
    elif hasattr(cfg, "partial_rotary_factor") and cfg.partial_rotary_factor is not None:
        partial_rotary_factor = float(cfg.partial_rotary_factor)
    elif isinstance(getattr(cfg, "rope_scaling", None), dict) and "partial_rotary_factor" in cfg.rope_scaling:
        partial_rotary_factor = float(cfg.rope_scaling["partial_rotary_factor"])

    dim = int(head_dim * partial_rotary_factor)
    if dim % 2 != 0:
        dim = dim - 1

    if rope_type == "default":
        if hasattr(mod, "compute_default_rope_parameters"):
            # Pass authoritative config and layer_type if supported
            inv_freq, attention_scaling = _call_with_supported_kwargs(
                mod.compute_default_rope_parameters, cfg, device=device, layer_type=layer_type
            )
        else:
            base = getattr(cfg, "rope_theta", 10000.0)
            if layer_rope_dict and "rope_theta" in layer_rope_dict:
                base = layer_rope_dict["rope_theta"]
            arange = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32)
            inv_freq = 1.0 / (base ** (arange / dim))
            attention_scaling = 1.0
        return inv_freq.to(dtype=torch.float32), attention_scaling

    # Scaled RoPE (yarn, linear, llama3) via transformers.modeling_rope_utils
    rope_utils = getattr(transformers, "modeling_rope_utils", None)
    rope_init_functions = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", {})
    if rope_type in rope_init_functions:
        init_fn = rope_init_functions[rope_type]
        inv_freq, attention_scaling = _call_with_supported_kwargs(
            init_fn, cfg, device=device, layer_type=layer_type
        )
        return inv_freq.to(dtype=torch.float32), attention_scaling

    # Fallback only for linear scaling if init_fn not registered in ROPE_INIT_FUNCTIONS
    if rope_type == "linear":
        base = getattr(cfg, "rope_theta", 10000.0)
        if layer_rope_dict and "rope_theta" in layer_rope_dict:
            base = layer_rope_dict["rope_theta"]
        rope_scaling = getattr(cfg, "rope_scaling", {}) or layer_rope_dict
        factor = float(rope_scaling.get("factor", 1.0))
        arange = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (arange / dim)) / factor
        return inv_freq.to(dtype=torch.float32), 1.0

    raise NotImplementedError(
        f"RoPE scaling type '{rope_type}' handler could not be located in transformers.modeling_rope_utils."
    )


def restore_fp32_rotary(model: torch.nn.Module) -> int:
    """
    Finds all RoPE rotary embedding modules across the model hierarchy and ensures their
    canonical inverse-frequency buffers are restored to torch.float32.

    Validates:
    - Output dtype is float32
    - Output shape matches existing buffer shape
    - Device placement is preserved
    - All generated values are finite
    - Original buffer persistence semantics are preserved

    If validation fails or model configuration is unsupported, leaves the buffer untouched.
    """
    restored_count = 0
    root_config = getattr(model, "config", None)
    if root_config is None and hasattr(model, "module"):
        root_config = getattr(model.module, "config", None)

    for _, mod in model.named_modules():
        cfg = getattr(mod, "config", root_config)
        # Match only validated RoPE buffer patterns on rotary modules
        matching_buffers: list[tuple[str, str | None]] = []
        for bname, _ in mod.named_buffers(recurse=False):
            is_rotary, layer_type = parse_rotary_buffer_name(mod, bname, config=cfg)
            if is_rotary:
                matching_buffers.append((bname, layer_type))

        if not matching_buffers:
            continue

        mod_buffers = list(mod.buffers())
        dev = mod_buffers[0].device if mod_buffers else torch.device("cpu")

        for bname, layer_type in matching_buffers:
            orig_buf = getattr(mod, bname, None)
            if orig_buf is None:
                continue

            try:
                inv_freq, attention_scaling = recompute_rotary_inv_freq(
                    mod, config=cfg, device=dev, layer_type=layer_type
                )
            except (NotImplementedError, ValueError) as e:
                logger.warning(
                    "Skipping float32 RoPE restoration for %s buffer '%s': %s. Module retains current precision.",
                    mod.__class__.__name__,
                    bname,
                    e,
                )
                continue

            # Strict validation before buffer replacement
            if inv_freq.dtype != torch.float32:
                logger.warning(
                    "Restoration validation failed for %s buffer '%s': expected float32, got %s.",
                    mod.__class__.__name__,
                    bname,
                    inv_freq.dtype,
                )
                continue

            if inv_freq.shape != orig_buf.shape:
                logger.warning(
                    "Restoration validation failed for %s buffer '%s': shape mismatch (expected %s, got %s).",
                    mod.__class__.__name__,
                    bname,
                    orig_buf.shape,
                    inv_freq.shape,
                )
                continue

            if not torch.all(torch.isfinite(inv_freq)):
                logger.warning(
                    "Restoration validation failed for %s buffer '%s': non-finite values detected.",
                    mod.__class__.__name__,
                    bname,
                )
                continue

            # Preserve device placement and persistence semantics
            target_buf = inv_freq.to(device=orig_buf.device, dtype=torch.float32)
            is_persistent = bname not in getattr(mod, "_non_persistent_buffers_set", set())
            mod.register_buffer(bname, target_buf, persistent=is_persistent)
            restored_count += 1

            if hasattr(mod, "attention_scaling") and attention_scaling is not None:
                mod.attention_scaling = float(attention_scaling)

    if restored_count > 0:
        logger.info(f"Restored {restored_count} RoPE rotary inverse-frequency buffers to torch.float32.")
    return restored_count
