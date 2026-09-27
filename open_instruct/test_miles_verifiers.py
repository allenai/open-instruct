"""MILES numeric equivalence must not change the shared verifier's scoring."""

import pytest

from open_instruct import ground_truth_utils
from open_instruct.miles.rewards import verifiers


@pytest.mark.parametrize(
    "prediction,label,expected,legacy",
    [
        ("The total cost is $96.00", "96", 1.0, 0.0),
        ("The answer is 42", "42.0", 1.0, 0.0),
        ("It takes 1,350 minutes", "1350", 1.0, 1.0),
        ("It takes 1350 minutes", "1,350", 1.0, 0.0),
        ("The total cost is $96.50", "96", 0.0, 0.0),
        ("About 0.33 of them", "0.3333", 0.0, 0.0),
        ("The answer is 7", "seven", 0.0, 0.0),
        ("No number", "no number", 1.0, 1.0),
    ],
)
def test_numeric_scoring_is_opt_in(prediction, label, expected, legacy):
    assert verifiers.GSM8KVerifier()([], prediction, label).score == expected
    assert ground_truth_utils.GSM8KVerifier()([], prediction, label).score == legacy
