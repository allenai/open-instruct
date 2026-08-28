"""Compatibility shims for search-branch ground_truth_utils symbols.

The Training Observatory was originally written on the search/BrowseComp fork,
whose open_instruct.ground_truth_utils exports BrowseComp-specific format
gates and a correctness-judge prompt. The terminal-RL branch does not have
those symbols; these fallbacks keep the shared rollout browser working with
no format gating. Terminal-specific outcome classification will replace the
BrowseComp logic over time.
"""

import re
import string

__all__ = [
    "BROWSECOMP_CORRECTNESS_JUDGE_PROMPT",
    "apply_browsecomp_format_gates",
    "apply_re_search_format_gates",
    "format_reference_answer",
    "normalize_answer",
]

BROWSECOMP_CORRECTNESS_JUDGE_PROMPT = (
    "Question: {question}\n\nReference answer: {reference_answer}\n\nResponse: {response}"
)


def apply_browsecomp_format_gates(terminal: str) -> tuple[str | None, str]:
    """No-op gate: never reports a format failure, inspects the stripped text."""
    return None, terminal.strip()


def apply_re_search_format_gates(terminal: str) -> tuple[str | None, str]:
    """No-op gate: never reports a format failure, inspects the stripped text."""
    return None, terminal.strip()


def format_reference_answer(value: object) -> str:
    return str(value)


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style).

    Copied from open_instruct.ground_truth_utils.normalize_answer: importing
    that module pulls in the full training stack (~50 s), which would dominate
    viewer startup for one small pure function.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))
