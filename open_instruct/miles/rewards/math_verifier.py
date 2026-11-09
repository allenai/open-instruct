"""MILES Math-Verify adapter with observable, bounded parsing and comparison.

References are answer strings, so wrap them as LaTeX; predictions retain the
library's answer extraction policy. Never fall back to the old permissive grader.
"""

import dataclasses
import threading
from itertools import product
from typing import Any

import math_verify
from math_verify import errors

from open_instruct import ground_truth_utils, logger_utils
from open_instruct.miles.rewards import rewards

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


class MathVerifier(ground_truth_utils.MathVerifier):
    """MILES math grading using pinned Hugging Face Math-Verify.

    Inherit the shared math configuration without adding another direct subclass
    to the legacy build_all_verifiers registry.
    """

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> ground_truth_utils.VerificationResult:
        diagnostics = {"symbolic_timeouts": 0}
        value = score(prediction, label, diagnostics)
        return ground_truth_utils.VerificationResult(score=value, diagnostics=diagnostics)

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> ground_truth_utils.VerificationResult:
        # SIGALRM requires a main thread. Reuse MILES's bounded subprocess pool
        # for direct async callers too; the worker invokes synchronous __call__.
        spec = {
            "factory": f"{type(self).__module__}.{type(self).__name__}",
            "config": dataclasses.asdict(self.verifier_config) if self.verifier_config is not None else {},
        }
        result = await rewards._IsolatedVerifier(spec).async_call(
            tokenized_prediction, prediction, label, query, rollout_state
        )
        return ground_truth_utils.VerificationResult(
            score=result.score,
            cost=result.cost,
            reasoning=getattr(result, "reasoning", None),
            diagnostics=result.diagnostics,
        )
