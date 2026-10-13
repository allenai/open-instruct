"""Require declared, validated serving arithmetic for MILES experiments."""

from open_instruct.miles.errors import InputError

MODE = "OLMO_SGLANG_CORE_COMPAT"
REQUIRED = "OLMO_SGLANG_REQUIRE_CORE_ROUNDING"
REASON = "OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON"
KEYS = {MODE, REQUIRED, REASON}


def apply(environment):
    """Set strong defaults and reject unacknowledged numerical overrides."""
    environment.setdefault(MODE, "rounding")
    environment.setdefault(REQUIRED, "1")
    if environment[REQUIRED] != "1":
        raise InputError(f"MILES requires {REQUIRED}=1; use {REASON} for a diagnostic override")
    if environment[MODE].strip().lower() != "rounding" and not environment.get(REASON, "").strip():
        raise InputError(f"MILES requires {MODE}=rounding; diagnostic overrides require a nonempty {REASON}")
