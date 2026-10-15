"""MILES numeric equivalence must not change the shared verifier's scoring."""

import subprocess
import sys

import pytest
from scripts.miles import development_eval

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
    row = {"text": prediction, "finish": {"type": "stop"}}
    assert development_eval.score_one(row, {"label": label, "verifier": "gsm8k"})["score"] == expected


@pytest.mark.parametrize(
    "prediction,label,expected",
    [
        ("18.00", "18", 1),
        ("18", "18.0", 1),
        ("1000", "1,000", 1),
        ("+18", "18", 1),
        ("-18.00", "-18", 1),
        ("18", "-18", 0),
        ("1,000,000", "1000000", 1),
        ("9007199254740993", "9007199254740992", 0),
        ("18.000000000000000001", "18", 0),
        ("18", "NaN", 0),
        ("18", "Infinity", 0),
        ("No number", "18", 0),
        ("SEVEN", "seven", 1),
        ("First 17, finally 18.00", "18", 1),
    ],
)
def test_development_gsm8k_matches_training_without_olmo_eval(prediction, label, expected):
    reward = verifiers.GSM8KVerifier()([], prediction, label).score
    assert reward == expected
    row = {"text": prediction, "finish": {"type": "stop"}}
    for gold in (label, [label], ["incorrect", label]):
        result = development_eval.score_one(row, {"label": gold, "verifier": "gsm8k"})
        assert result["score"] == reward


@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_development_nonfinite_values_are_not_numbers(value):
    assert development_eval._exact_number(value) is None


def test_development_numeric_equivalence_preserves_completion_gates():
    item = {"label": ["17", "18"], "verifier": "gsm8k"}
    for text, finish, completed in [
        ("</think> <answer>18.00</answer>", "stop", 1),
        ("</think> <answer>18.00</answer>", "length", 0),
        ("18.00", "stop", 0),
    ]:
        result = development_eval.score_one({"text": text, "finish": {"type": finish}}, item)
        assert result["score"] == 1
        assert result["completed_final_score"] == completed
        assert result["scoring"]["extracted"] == "18.00"


def test_development_gsm8k_runs_with_only_the_standard_library():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import runpy, sys; scorer = runpy.run_path(sys.argv[1]); "
            "assert scorer['score_one']({'text': '18.00', 'finish': {'type': 'stop'}}, "
            "{'label': '18', 'verifier': 'gsm8k'})['score'] == 1",
            development_eval.__file__,
        ],
        check=True,
    )
