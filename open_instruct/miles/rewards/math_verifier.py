"""MILES Math-Verify adapter with observable, bounded parsing and comparison.

References are answer strings, so wrap them as LaTeX; predictions retain the
library's anchored answer extraction policy. Never fall back to the old
permissive grader.
"""

import dataclasses
import functools
import re
import threading
from itertools import product
from typing import Any

import math_verify
from math_verify import errors
from sympy import Eq

from open_instruct import ground_truth_utils, logger_utils, math_utils
from open_instruct.miles.rewards import rewards

logger = logger_utils.setup_logger(__name__)
_TIMEOUT_SECONDS = 5
# Unwrap paired Markdown bold spans without deleting Python-style powers such
# as 2**3. Word/star boundaries also avoid matching inside longer star runs.
_BOLD = re.compile(r"(?<![\w*])\*\*(?!\*)(?=\S)(.+?)(?<=\S)\*\*(?![\w*])", re.DOTALL)


# One math-mode delimiter pair around an entire answer, e.g. "\(2^4\)" or "$\pi$". The inner
# text may not contain the same delimiter, so "\(a\) and \(b\)" and other multi-span or prose
# answers are left unchanged, and an escaped dollar ("\$5") is never treated as a delimiter.
_OUTER_MATH_DELIMITERS = (
    re.compile(r"\\\((?P<inner>(?:(?!\\[()]).)*)\\\)", re.DOTALL),
    re.compile(r"\\\[(?P<inner>(?:(?!\\[\[\]]).)*)\\\]", re.DOTALL),
    re.compile(r"\$\$(?P<inner>[^$]*)\$\$", re.DOTALL),
    re.compile(r"\$(?P<inner>[^$]*)(?<!\\)\$", re.DOTALL),
)


def strip_outer_math_delimiters(text: str) -> str:
    """Remove one outer \\(...\\), \\[...\\], $$...$$ or $...$ pair enclosing the whole answer.

    Returns the input unchanged when no single pair spans all of it, the content would be
    empty, or the input is not a string.
    """
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    for pattern in _OUTER_MATH_DELIMITERS:
        match = pattern.fullmatch(stripped)
        if match and match.group("inner").strip():
            return match.group("inner").strip()
    return text


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


@functools.lru_cache(maxsize=1)
def _false_equation():
    sentinel = math_verify.parse(
        r"\boxed{1 + 2 = 8}",
        extraction_config=(math_verify.LatexExtractionConfig(),),
        parsing_timeout=_TIMEOUT_SECONDS,
        raise_on_error=True,
    )
    if not sentinel:
        logger.warning("Math-Verify could not parse the fixed false-equation sentinel")
        raise ValueError("Math-Verify could not parse the false-equation sentinel")
    return sentinel


@functools.lru_cache(maxsize=4096)
def _requires_legacy_verifier(reference: str) -> bool:
    """Cache whether this reference accepts a known-wrong equation.

    Math-Verify 0.9.0 can match unrelated equations when its solver returns two
    empty lists. Such references use legacy comparison rules and the literal-string path.
    Probe equation candidates only: a scalar 8 legitimately matches an equation
    with RHS 8 under the library's answer-extraction rules.
    Cache only the decision, not parsed references; exceptions are never cached.
    """
    gold = math_verify.parse(
        r"\boxed{" + reference + "}",
        extraction_config=(math_verify.LatexExtractionConfig(),),
        parsing_timeout=_TIMEOUT_SECONDS,
        raise_on_error=True,
    )
    equations = [value for value in gold if isinstance(value, Eq)]
    return bool(equations) and math_verify.verify(
        equations, _false_equation(), timeout_seconds=_TIMEOUT_SECONDS, raise_on_error=True
    )


def _legacy_compare(answer: str, reference: str, diagnostics: dict[str, int]) -> bool:
    """Use legacy expression comparison while preserving every equation side."""
    expected_parts = reference.split("=")
    answer_parts = answer.split("=")
    if len(expected_parts) != len(answer_parts):
        return False
    return all(
        math_utils.is_equiv(actual, expected, diagnostics=diagnostics)
        or math_utils.hendrycks_is_equiv(actual, expected)
        for actual, expected in zip(answer_parts, expected_parts)
    )


def score(prediction: str, label: str, diagnostics: dict[str, int]) -> float:
    """Score on a process main thread; async callers must use process isolation."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Math-Verify requires a main thread; use MathVerifier.async_call for async grading")
    # Strip only a single enclosing span. Multi-span references must reach the
    # parser intact, e.g. \(x=2\) or \(x=-2\).
    reference = strip_outer_math_delimiters(label).strip()
    if not reference or not prediction.strip():
        return 0.0
    gold = _parse(r"\boxed{" + reference + "}", (math_verify.LatexExtractionConfig(),), diagnostics)
    if not gold:
        return 0.0
    # Boxes remain eligible, but incidental numbers/LaTeX in unfinished reasoning
    # must not become answers. Strip formatting so **Answer:** still anchors.
    target = _parse(
        _BOLD.sub(r"\1", prediction),
        (
            math_verify.ExprExtractionConfig(try_extract_without_anchor=False),
            math_verify.LatexExtractionConfig(try_extract_without_anchor=False),
        ),
        diagnostics,
    )
    if not target:
        return 0.0
    use_legacy = False
    try:
        if any(isinstance(value, Eq) for value in gold):
            use_legacy = _requires_legacy_verifier(reference)
        string_only = use_legacy
    except errors.TimeoutException:
        _record_timeout(diagnostics)
        string_only = True
    except Exception:
        logger.debug("Math-Verify reference probe failed; using literal-string comparison")
        string_only = True
    if use_legacy:
        # Reuse the old comparison rules on already-anchored candidates. The old
        # extractor/normalizer can discard equation LHSs (e.g. 1=0 -> 0), so do
        # not rerun it on the full response or compare whole equations at once.
        for answer in target:
            if isinstance(answer, str) and _legacy_compare(answer, reference, diagnostics):
                return 1.0
    if string_only:
        # A failed probe cannot certify symbolic comparison either. Preserve
        # exact extracted-string matches; Math-Verify symbolic matches are disabled.
        gold = [value for value in gold if isinstance(value, str)]
        target = [value for value in target if isinstance(value, str)]
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
