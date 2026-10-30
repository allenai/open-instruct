"""Configuration shared by preparation and the fixed source-mixture provider."""

import math
import re

from open_instruct.miles.errors import InputError


def validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise InputError("data.sources must be a nonempty list")
    names, fractions = [], []
    for source in sources:
        if not isinstance(source, dict):
            raise InputError("Each data.sources entry must be a table")
        name, fraction = source.get("name"), source.get("fraction")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise InputError("Source names must be unique safe identifiers (letters, digits, underscore, dash)")
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise InputError(f"Source {name}: fraction must be finite and in (0, 1]")
        names.append(name)
        fractions.append(fraction)
    if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise InputError(f"Source fractions must sum to 1, got {math.fsum(fractions)}; fractions are not normalized")


def source_data(source, data):
    return {"seed": data.get("seed", 17), "shuffle": data.get("shuffle", True)} | {
        key: value for key, value in source.items() if key not in ("name", "fraction")
    }
