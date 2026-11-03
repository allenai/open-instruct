import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.rewards import nvidia_constraints as nv


def score(verifier, text, label):
    return asyncio.run(verifier.async_call([], text, label)).score


def test_calendar_boundaries_and_overlap():
    label = {
        "0": {"duration": 30, "min_time": "10:00", "max_time": "16:00", "constraint": "before 11am"},
        "1": {"duration": 60, "min_time": "10:00", "max_time": "16:00", "constraint": "after 11am"},
    }
    answer = [
        {"event_id": 0, "start_time": "10:30", "duration": 30},
        {"event_id": 1, "start_time": "11:00", "duration": 60},
    ]
    verifier = nv.CalendarVerifier()
    assert score(verifier, "<think>reason</think>" + json.dumps(answer), label) == 1
    for bad in [[], answer[:1], answer + answer[:1], {}, None]:
        assert score(verifier, json.dumps(bad), label) == 0
    for key, value in [("duration", True), ("duration", 31), ("start_time", "10:31"), ("start_time", "25:00")]:
        wrong = copy.deepcopy(answer)
        wrong[0][key] = value
        assert score(verifier, json.dumps(wrong), label) == 0
    overlap = copy.deepcopy(label)
    overlap["1"]["constraint"] = None
    answer[1]["start_time"] = "10:45"
    assert score(verifier, json.dumps(answer), overlap) == 0


def test_calendar_invalid_target_is_not_reward_zero():
    with pytest.raises(ValueError):
        score(nv.CalendarVerifier(), "[]", {})


def test_nemo_native_parity_and_final_region():
    pytest.importorskip("verifiable_instructions")
    label = {
        "family": "nemo",
        "target": {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["cat"]}, {}],
        },
    }
    verifier = nv.InstructionVerifier()
    for text in ["cat", "dog", "cat, dog", "A cat sleeps."]:
        _, compiled = nv.instruction_contract(json.dumps(label))
        expected = float(all(checker.check_following(text) for _, checker, _ in compiled))
        assert score(verifier, text, label) == expected
    assert score(verifier, "<think>cat</think>dog", label) == 0
    assert score(verifier, "<think>dog</think>cat", label) == 1
    assert score(verifier, "", label) == 0


def test_nemo_rejects_arguments_that_native_would_repair():
    pytest.importorskip("verifiable_instructions")
    label = {
        "family": "nemo",
        "target": {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 0, "relation": "at least"}],
        },
    }
    with pytest.raises(ValueError, match="Nonpositive"):
        nv.instruction_contract(json.dumps(label))


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
    assert nv.check_repeat_phrase(text, "Live each day to the fullest", 2) is expected


def test_repeat_phrase_endpoints_are_literal():
    assert nv.check_repeat_phrase("A+ replacement Z?", "A+ middle Z?", 1) is True
    assert nv.check_repeat_phrase("AA replacement Z", "A+ middle Z?", 1) is False


def test_nemo_repeat_phrase_regressions():
    pytest.importorskip("verifiable_instructions")
    label = {
        "family": "nemo",
        "target": {
            "instruction_id_list": ["copy:repeat_phrase", "punctuation:no_comma"],
            "kwargs": [{"phrase": "Live each day to the fullest", "small_n": 2}, {}],
        },
    }
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
