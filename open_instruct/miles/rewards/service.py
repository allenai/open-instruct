"""Share setting parsing and reusable HTTP sessions between external reward clients.
These helpers give code execution and language-model judges consistent precedence
for explicit arguments and environment values, and construct connection pools with
caller-selected retry behavior. They keep service plumbing separate from each
verifier's prompt construction and scoring rules.
"""

import functools
import math
import os
from typing import Any

import requests


def number_setting(args: Any, attribute: str, environment: str, default: int | float) -> int | float:
    value = getattr(args, attribute, None)
    if value is None:
        value = os.environ.get(environment, default)
    expected_type = int if isinstance(default, int) else float
    try:
        parsed = expected_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{environment} must be a finite number") from error
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise ValueError(f"{environment} must be a finite number")
    return parsed


def bool_setting(args: Any, attribute: str, environment: str, default: bool) -> bool:
    value = getattr(args, attribute, None)
    if value is None:
        value = os.environ.get(environment, str(default))
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{environment} must be a boolean")


@functools.lru_cache(maxsize=8)
def http_session(*, pool_size: int, retries=0) -> requests.Session:
    """Reuse a session per transport policy; code and judge retry policies stay distinct."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
