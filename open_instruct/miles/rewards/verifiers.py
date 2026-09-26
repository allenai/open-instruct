"""Verifier variants selected by MILES datasets without changing legacy scoring."""

import re
from decimal import Decimal, InvalidOperation

from open_instruct import ground_truth_utils


def _exact_number(text: str) -> Decimal | None:
    """Parse finite decimal values exactly, allowing thousands separators."""
    try:
        value = Decimal(str(text).strip().replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


class GSM8KVerifier(ground_truth_utils.GSM8KVerifier):
    """Compare the last answer numerically (e.g. 96.00 equals 96), without rounding."""

    def __call__(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)
        found = _exact_number(numbers[-1]) if numbers else None
        expected = _exact_number(label)
        if found is not None and expected is not None:
            return ground_truth_utils.VerificationResult(score=float(found == expected))
        return super().__call__(tokenized_prediction, prediction, label, query, rollout_state)
