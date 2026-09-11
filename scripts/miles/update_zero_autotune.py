"""Diagnostic-only replay of declared Triton configurations, before model warmup.

The reference selects one existing candidate for each named tuner. Other tuners
remain unchanged. Actual Python tuner invocations during armed eager prefill are
recorded; CUDA graph replay and unwrapped kernels are outside this observation.
"""

import functools
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from triton.runtime.autotuner import Autotuner

PIN_ENV = "OI_UPDATE_ZERO_AUTOTUNE_REFERENCE"
PREFIXES = ("fla.", "sglang.kernels.ops.attention.fla.", "sglang.srt.batch_invariant_ops.")
_STATE = {"profile": None, "capture": None}


def configuration(config):
    return {
        "kwargs": dict(config.kwargs),
        **{name: getattr(config, name, None) for name in ("num_warps", "num_stages", "num_ctas", "maxnreg")},
    }


def discover_tuners():
    found, seen = {}, set()
    for module_name, module in sorted(list(sys.modules.items())):
        if module is None or not module_name.startswith(PREFIXES):
            continue
        for symbol, candidate in sorted(list(vars(module).items())):
            for _ in range(8):
                if isinstance(candidate, Autotuner):
                    break
                candidate = getattr(candidate, "fn", None)
                if candidate is None:
                    break
            if isinstance(candidate, Autotuner) and id(candidate) not in seen:
                seen.add(id(candidate))
                found[module_name + "." + symbol] = candidate
    return found


def apply_reference(path, *, tuners=None):
    """Validate every requested candidate before mutating any tuner; no tuning."""
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if _STATE["profile"] is not None:
        if _STATE["profile"]["sha256"] != sha:
            raise ValueError("Cannot replace an installed autotune reference")
        return _STATE["profile"]
    profile = json.loads(raw)
    if (
        profile.get("schema_version") != 1
        or not isinstance(profile.get("configurations"), dict)
        or not profile["configurations"]
    ):
        raise ValueError("Expected nonempty schema-1 configurations")
    if tuners is None:
        for name in profile["configurations"]:
            if not name.startswith(PREFIXES) or "." not in name:
                raise ValueError(f"Unsupported tuner module: {name}")
            importlib.import_module(name.rsplit(".", 1)[0])
        tuners = discover_tuners()
    missing = set(profile["configurations"]) - tuners.keys()
    if missing:
        raise ValueError(f"Reference tuners not loaded: {sorted(missing)}")
    selected = {}
    for name, desired in profile["configurations"].items():
        candidates = [candidate for candidate in tuners[name].configs if configuration(candidate) == desired]
        if not candidates:
            raise ValueError(f"Reference config is not a declared candidate: {name}")
        if tuners[name].cache:
            raise ValueError(f"Cannot pin an already executed tuner: {name}")
        selected[name] = candidates[0]
    report = {
        "sha256": sha,
        "path": str(path),
        "reference": profile.get("reference"),
        "pinned_configurations": profile["configurations"],
        "observed_unpinned_tuners": sorted(set(tuners) - selected.keys()),
        "interpretation": "Singleton declared configurations installed before warmup. Invocation records cover wrapped Python tuner calls during armed prefill, not graph replay or all GPU kernels.",
    }
    for name, tuner in tuners.items():
        pinned = name in selected
        if pinned:
            tuner.configs = [selected[name]]
        original = tuner.run

        @functools.wraps(original)
        def run(*args, _original=original, _tuner=tuner, _name=name, _pinned=pinned, **kwargs):
            result = _original(*args, **kwargs)
            active = _STATE["capture"]
            if active is not None:
                applied = configuration(_tuner.best_config)
                if _pinned and applied != profile["configurations"][_name]:
                    raise ValueError(f"Pinned configuration drifted: {_name}")
                active["invocations"].append({"kernel": _name, "pinned": _pinned, "configuration": applied})
            return result

        tuner.run = run
    _STATE["profile"] = report
    return report


def apply_from_environment():
    path = os.environ.get(PIN_ENV)
    return apply_reference(path) if path else None


def begin_capture(capture_id):
    if _STATE["profile"] is not None:
        if _STATE["capture"] is not None:
            raise ValueError("Nested tuner capture")
        _STATE["capture"] = {"capture_id": capture_id, "invocations": []}


def finish_capture():
    active = _STATE["capture"]
    _STATE["capture"] = None
    if active is None:
        return None
    active["profile"] = _STATE["profile"]
    active["pinned_tuners_without_observed_invocation"] = sorted(
        set(_STATE["profile"]["pinned_configurations"]) - {row["kernel"] for row in active["invocations"]}
    )
    return active
