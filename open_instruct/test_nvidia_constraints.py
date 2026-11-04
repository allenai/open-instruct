"""Check native IF semantics and calendar violations through runtime adapters."""

import asyncio
import json

import pytest

from open_instruct.miles.rewards import nvidia_constraints as native


def target(name, args, **extra):
    return {"family": "nemo", "target": {"instruction_id_list": [name], "kwargs": [args], **extra}}


@pytest.mark.parametrize(
    "name,args,gold,wrong",
    [
        ("count:lowercase_counting", {"N": 2}, "one two three four", "one one one"),
        ("letters:letter_counting", {"N": 4, "relation": "less than"}, "cat", "中文中文"),
        ("letters:letter_counting", {"N": 4, "relation": "at least"}, "中文中文", "cat"),
        ("keywords:existence", {"keywords": ["apple"]}, "an apple", "a pear"),
        ("punctuation:no_comma", {}, "one two", "one, two"),
        ("detectable_format:json_format", {}, '{"a":1}', "hello"),
    ],
)
def test_native_controls(name, args, gold, wrong):
    label = target(name, args)
    verifier = native.InstructionVerifier()
    assert verifier([], gold, label).score == 1
    assert verifier([], wrong, label).score == 0
    assert verifier([], "", label).score == 0
    assert verifier([], "thinking</think>" + gold, label).score == 1


def test_grading_mode_and_argument_repairs():
    verifier = native.InstructionVerifier()
    label = target("keywords:existence", {"keywords": ["apple"]}, grading_mode="fraction")
    label["target"]["instruction_id_list"].append("punctuation:no_comma")
    label["target"]["kwargs"].append({})
    assert verifier([], "an apple,", label).score == 0.5
    label["target"]["grading_mode"] = "all"
    assert verifier([], "an apple,", label).score == 0
    for invalid in [
        target("keywords:frequency", {}),
        target("count:lowercase_counting", {"N": 0}),
        target("letters:letter_counting", {"N": 2, "relation": "less than"}),
        target("language:response_language", {"language": "en"}),
        target("punctuation:no_comma", {}, grading_mode="typo"),
    ]:
        with pytest.raises(ValueError):
            verifier([], "apple", invalid)


def test_calendar_full_schedule_and_near_misses():
    verifier = native.CalendarVerifier()
    label = {
        "0": {"duration": 60, "min_time": "9am", "max_time": "5pm", "constraint": "before 11am"},
        "1": {"duration": 30, "min_time": "9am", "max_time": "5pm", "constraint": "after 10am"},
    }
    events = [
        {"event_id": 0, "start_time": "09:00", "duration": 60},
        {"event_id": 1, "start_time": "10:00", "duration": 30},
    ]
    assert verifier([], json.dumps(events), label).score == 1
    for changes in [{"event_id": 0}, {"start_time": "9:30"}, {"duration": 31}, {"start_time": "17:00"}]:
        assert verifier([], json.dumps([events[0], {**events[1], **changes}]), label).score == 0
    assert verifier([], "[]", {}).score == 1
    assert verifier([], "", {}).score == 0
    assert verifier([], json.dumps(events[:1]), label).score == 0
    with pytest.raises(ValueError, match="Unsatisfiable"):
        verifier([], "[]", {"0": {**label["0"], "constraint": "at 5pm"}})


def test_actual_worker_path():
    label = target("count:lowercase_counting", {"N": 2})
    assert asyncio.run(native.InstructionVerifier().async_call([], "one two three", label)).score == 1
    assert asyncio.run(native.CalendarVerifier().async_call([], "[]", {})).score == 1
