"""Require declared, validated serving arithmetic for MILES experiments."""

import re

from open_instruct.miles.errors import InputError

MODE = "OLMO_SGLANG_CORE_COMPAT"
REQUIRED = "OLMO_SGLANG_REQUIRE_CORE_ROUNDING"
REASON = "OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON"
PROFILE = "OLMO_SGLANG_KERNEL_PROFILE"
PROFILE_MODE = "OLMO_SGLANG_KERNEL_PROFILE_MODE"
PROFILE_DIGEST = "OLMO_SGLANG_KERNEL_PROFILE_SHA256"
PROFILE_REPORT = "OLMO_SGLANG_KERNEL_PROFILE_REPORT"
KEYS = {MODE, REQUIRED, REASON, PROFILE, PROFILE_MODE, PROFILE_DIGEST, PROFILE_REPORT}


def apply(environment):
    """Set strong defaults and reject unacknowledged numerical overrides."""
    environment.setdefault(MODE, "rounding")
    environment.setdefault(REQUIRED, "1")
    if environment[REQUIRED] != "1":
        raise InputError(f"MILES requires {REQUIRED}=1; use {REASON} for a diagnostic override")
    if environment[MODE].strip().lower() != "rounding" and not environment.get(REASON, "").strip():
        raise InputError(f"MILES requires {MODE}=rounding; diagnostic overrides require a nonempty {REASON}")

    if not KEYS.intersection(environment) - {MODE, REQUIRED, REASON}:
        return
    path = environment.get(PROFILE, "")
    selected = environment.setdefault(PROFILE_MODE, "strict" if path else "off")
    reason = environment.get(REASON, "").strip()
    if selected not in {"strict", "record", "off"}:
        raise InputError(f"Unknown {PROFILE_MODE}: {selected}")
    if selected in {"strict", "record"} and not path.startswith("/"):
        raise InputError(f"{PROFILE} must be an absolute shared-mount or image path")
    if selected == "strict" and not re.fullmatch(r"[0-9a-f]{64}", environment.get(PROFILE_DIGEST, "")):
        raise InputError(f"Strict kernel profiles require a lowercase content SHA256 in {PROFILE_DIGEST}")
    if (selected == "record" or (selected == "off" and path)) and not reason:
        raise InputError(f"Recording or disabling a configured kernel profile requires {REASON}")
