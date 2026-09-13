"""Resolve explicit SGLang graph settings for CPU-side checks, without defaults.

The pinned parser gives JSON overrides precedence over convenience flags, which
in turn override legacy disable switches. Model-dependent defaults stay unknown.
"""

import json

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError


def explicit_settings(options):
    result = {phase: {} for phase in ("decode", "prefill")}
    for phase, settings in result.items():
        if options.get("sglang_disable_cuda_graph") or options.get(f"sglang_disable_{phase}_cuda_graph"):
            settings["backend"] = "disabled"
        for field, flag in (("backend", "backend"), ("max_bs", "max_bs"), ("bs", "bs")):
            value = options.get(f"sglang_cuda_graph_{flag}_{phase}")
            if value is not None:
                settings[field] = value
    raw = options.get("sglang_cuda_graph_config")
    if raw is not None:
        try:
            raw = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError as error:
            raise InputError(
                "miles.sglang_cuda_graph_config must contain valid JSON or a TOML inline table"
            ) from error
        validation.mapping(raw, "miles.sglang_cuda_graph_config")
        for phase, settings in raw.items():
            if phase not in result:
                raise InputError(f"Unknown CUDA graph phase {phase!r}; use decode or prefill")
            validation.mapping(settings, f"miles.sglang_cuda_graph_config.{phase}")
            result[phase].update(settings)
    for phase, settings in result.items():
        prefix = f"miles.sglang_cuda_graph_config.{phase}"
        if settings.get("backend") is not None:
            validation.choice(
                settings["backend"], prefix + ".backend", ("disabled", "full", "breakable", "tc_piecewise")
            )
        if settings.get("max_bs") is not None:
            validation.integer(settings["max_bs"], prefix + ".max_bs", minimum=1)
        if settings.get("bs") is not None:
            if not isinstance(settings["bs"], list):
                raise InputError(prefix + ".bs must be a list of positive batch sizes")
            for value in settings["bs"]:
                validation.integer(value, prefix + ".bs entry", minimum=1)
    return result
