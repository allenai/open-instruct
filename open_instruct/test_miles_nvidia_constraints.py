"""Regression cases from the original NVIDIA trial verifier integration."""

import json
from types import SimpleNamespace

import pytest

from open_instruct.IFEvalG import instructions_registry
from open_instruct.miles.rewards import nvidia_constraints as nv


def score(verifier, text, label):
    return verifier([], text, label).score


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Live every day to the fullest\nLive each night to the fullest", True),
        ("Live each day to the fullest\nLive each day to the fullest", False),
        ("Live each day to the fullest\nLive each night to the fullest", False),
        ("Live each night to the fullest\nLive each day to the fullest", False),
        ("Live every night to the fullest\nLive each night to the fullest", False),
        ("Live each night to the fullest", False),
        ("Live each night to the fullest\n" * 3, False),
        ("Live each very long night to the fullest\nLive each night to the fullest", False),
        ("Enjoy each day to the fullest\nLive each night to the fullest", False),
        ("Live each day to the maximum\nLive each night to the fullest", False),
        ("", False),
    ],
)
def test_repeat_phrase_requires_one_interior_change_in_every_repetition(text, expected):
    checker = instructions_registry.INSTRUCTION_DICT["copy:repeat_phrase"]("copy:repeat_phrase")
    checker.build_description(phrase="Live each day to the fullest", small_n=2)
    assert checker.check_following(text) is expected


def test_nemo_repeat_phrase_regressions():
    label = {
        "family": "nemo",
        "target": {
            "instruction_id_list": ["copy:repeat_phrase", "punctuation:no_comma"],
            "kwargs": [{"phrase": "Live each day to the fullest", "small_n": 2}, {}],
        },
    }
    _, compiled = nv.instruction_contract(json.dumps(label))
    assert type(compiled[0][1]) is instructions_registry.INSTRUCTION_DICT["copy:repeat_phrase"]
    verifier = nv.InstructionVerifier()
    exact = "Live each day to the fullest"
    changed = "Live each night to the fullest"
    for text in [f"{exact}\n{exact}", f"{exact}\n{changed}", f"{changed}\n{exact}"]:
        assert score(verifier, text, label) == 0
    assert score(verifier, f"{changed}\n{changed}", label) == 1
    label["target"]["grading_mode"] = "fraction"
    assert score(verifier, f"{exact}\n{exact}", label) == 0.5
    label["target"]["kwargs"][0]["phrase"] = "two words"
    with pytest.raises(ValueError, match="interior word"):
        nv.instruction_contract(json.dumps(label))


def test_nemo_unexpected_checker_return_still_raises_with_identity(monkeypatch):
    checker = SimpleNamespace(check_following=lambda _: None)
    monkeypatch.setattr(nv, "instruction_contract", lambda _: ("nemo", [("unexpected:checker", checker, {})]))
    with pytest.raises(ValueError, match="unexpected:checker.*NoneType"):
        score(nv.InstructionVerifier(), "answer", {"target": {}})
