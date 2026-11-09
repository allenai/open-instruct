"""Math-Verify adapter with observable, bounded parsing and comparison.

References are answer strings, so wrap them as LaTeX; predictions retain the
library's answer extraction policy. Never fall back to the old permissive grader.
"""

import threading
from itertools import product

import math_verify
from math_verify import errors

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)
_TIMEOUT_SECONDS = 5


def _record_timeout(diagnostics: dict[str, int]) -> None:
    diagnostics["symbolic_timeouts"] += 1
    logger.warning("Math-Verify parsing or comparison exceeded %d seconds", _TIMEOUT_SECONDS)


def _parse(text: str, configs: tuple, diagnostics: dict[str, int]) -> list:
    try:
        return math_verify.parse(
            text, extraction_config=configs, parsing_timeout=_TIMEOUT_SECONDS, raise_on_error=True
        )
    except errors.TimeoutException:
        _record_timeout(diagnostics)
    except Exception:
        # Match Math-Verify's default parse-failure behavior, without treating
        # an ordinary parse error as a timeout or logging untrusted answers.
        logger.debug("Math-Verify could not parse an answer")
    return []


def score(prediction: str, label: str, diagnostics: dict[str, int]) -> float:
    """Score on a process main thread; async callers must use process isolation."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Math-Verify requires a main thread; use MathVerifier.async_call for async grading")
    reference = label.strip()
    # Dataset references often already carry math delimiters. Remove just the
    # outer wrapper before boxing, retaining commas and internal structure.
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
        if reference.startswith(left) and reference.endswith(right) and len(reference) >= len(left) + len(right):
            reference = reference[len(left) : -len(right)].strip()
            break
    if not reference or not prediction.strip():
        return 0.0
    gold = _parse(r"\boxed{" + reference + "}", (math_verify.LatexExtractionConfig(),), diagnostics)
    if not gold:
        return 0.0
    target = _parse(prediction, (math_verify.ExprExtractionConfig(), math_verify.LatexExtractionConfig()), diagnostics)
    # Reproduce verify(list, list)'s candidate order and exception recovery, but
    # expose timeout counts. The library also supplies literal-string candidates;
    # those may still agree after a symbolic comparison times out.
    for expected, answer in product(gold, target):
        try:
            if math_verify.verify(expected, answer, timeout_seconds=_TIMEOUT_SECONDS, raise_on_error=True):
                return 1.0
        except errors.TimeoutException:
            _record_timeout(diagnostics)
        except Exception:
            logger.debug("Math-Verify could not compare an answer pair")
    return 0.0
