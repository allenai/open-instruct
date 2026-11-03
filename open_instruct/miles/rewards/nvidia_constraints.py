"""Portable calendar and pinned NeMo instruction-following contracts."""

import importlib
import json
import re
from functools import lru_cache

from open_instruct import ifeval_constraints
from open_instruct.miles.rewards import candidate_verifiers as common
from open_instruct.miles.rewards import task_verifiers


def minutes(value):
    if not isinstance(value, str):
        raise ValueError("Time must be a string")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", value.strip())
    if not match:
        raise ValueError("Unsupported time")
    hour, minute, period = int(match[1]), int(match[2] or 0), match[3]
    if minute >= 60 or (period and not 1 <= hour <= 12) or (not period and hour >= 24):
        raise ValueError("Time out of range")
    return ((hour % 12) + (12 if period == "pm" else 0) if period else hour) * 60 + minute


def calendar_contract(target):
    if not isinstance(target, dict) or not target:
        raise ValueError("Empty calendars need a separate abstention contract")
    result = {}
    for key, event in target.items():
        duration = event["duration"]
        if type(duration) is not int or duration <= 0:
            raise ValueError("Calendar duration must be positive")
        lower, upper = minutes(event["min_time"]), minutes(event["max_time"]) - duration
        constraint = event["constraint"]
        if constraint is not None:
            if not isinstance(constraint, str):
                raise ValueError("Invalid calendar constraint")
            if constraint.startswith("at "):
                lower = max(lower, minutes(constraint[3:]))
                upper = min(upper, minutes(constraint[3:]))
            elif constraint.startswith("before "):
                upper = min(upper, minutes(constraint[7:]) - duration)
            elif constraint.startswith("after "):
                lower = max(lower, minutes(constraint[6:]))
            elif constraint.startswith("between "):
                first, last = constraint[8:].split(" and ")
                lower, upper = max(lower, minutes(first)), min(upper, minutes(last) - duration)
            else:
                raise ValueError("Unknown calendar constraint")
        if lower > upper:
            raise ValueError("Unsatisfiable calendar window")
        result[str(key)] = (lower, upper, duration)
    return result


class CalendarVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        contract = calendar_contract(label)
        try:
            events = common.strict_json(common.answer_text(prediction))
            if not isinstance(events, list) or len(events) != len(contract):
                return common.Result(0.0)
            intervals, seen = [], set()
            for event in events:
                key = str(event["event_id"])
                if key not in contract or key in seen:
                    return common.Result(0.0)
                seen.add(key)
                lower, upper, duration = contract[key]
                if type(event["duration"]) not in (int, float) or event["duration"] != duration:
                    return common.Result(0.0)
                start = minutes(event["start_time"])
                if not lower <= start <= upper:
                    return common.Result(0.0)
                intervals.append((start, start + duration))
            intervals.sort()
            valid = all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
            return common.Result(float(valid))
        except (ValueError, TypeError, KeyError):
            return common.Result(0.0)


@lru_cache(maxsize=4096)
def instruction_contract(serialized):
    label = json.loads(serialized)
    family, target = label["family"], label["target"]
    if family != "nemo":
        raise ValueError("Only the pinned NeMo instruction family is supported")
    modules = {"nemo": "verifiable_instructions.instructions_registry"}
    registry = importlib.import_module(modules[family]).INSTRUCTION_DICT
    ids = target["instruction_id_list"]
    arguments = target["kwargs"]
    issues = ifeval_constraints.if_training_issues({"instruction_id": ids, "kwargs": arguments})
    if issues:
        raise ValueError(f"Unsuitable IF training target: {issues}")
    if not ids or len(ids) != len(arguments):
        raise ValueError("Empty or unaligned IF constraints")
    compiled = []
    for name, args in zip(ids, arguments, strict=True):
        if name in {"language:response_language", "change_case:english_capital", "change_case:english_lowercase"}:
            raise ValueError("Language classification needs separate calibration")
        if name in {"letters:letter_counting", "count:lowercase_counting"}:
            raise ValueError("Pinned NeMo checker predates the reviewed counting fixes")
        checker = registry[name](name)
        args = {k: v for k, v in (args or {}).items() if v is not None}
        if set(args) != set(checker.get_instruction_args_keys()):
            raise ValueError(f"Missing or unexpected arguments for {name}")
        for key, value in args.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, list)):
                raise ValueError(f"Unsupported argument {key}")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Empty argument {key}")
            if isinstance(value, list) and (not value or not all(isinstance(v, str) and v.strip() for v in value)):
                raise ValueError(f"Malformed argument {key}")
            if isinstance(value, int) and (
                value < 0 or value == 0 and not (name == "new:copy_span_idx" and key in {"n_start", "n_end"})
            ):
                raise ValueError(f"Nonpositive argument {key}; qualify separately")
        if name == "length_constraints:nth_paragraph_first_word" and args["nth_paragraph"] > args["num_paragraphs"]:
            raise ValueError("Unsatisfiable paragraph position")
        try:
            checker.build_description(**args)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"Invalid native checker arguments for {name}") from exc
        # Native generators repair invalid arguments, sometimes randomly.
        # Training targets must retain their explicit supplied parameters.
        expected_args = dict(args)
        if name == "keywords:existence":
            expected_args["keywords"] = sorted(args["keywords"])
        elif name == "keywords:forbidden_words":
            expected_args["forbidden_words"] = sorted(set(args["forbidden_words"]))
        if (checker.get_instruction_args() or {}) != expected_args:
            raise ValueError(f"Native checker changed target arguments for {name}")
        compiled.append((name, checker, args))
    return family, compiled


class InstructionVerifier(CalendarVerifier):
    async def async_call(self, tokens, prediction, label, **kwargs):
        family, compiled = instruction_contract(json.dumps(label, sort_keys=True, allow_nan=False))
        final = common.answer_text(prediction)
        if not final:
            return common.Result(0.0)
        passed = []
        for _name, checker, _args in compiled:
            value = checker.check_following(final)
            if type(value) is not bool:
                raise ValueError("IF checker did not return a Boolean")
            passed.append(value)
        fraction = label.get("target", {}).get("grading_mode") == "fraction"
        return common.Result(sum(passed) / len(passed) if fraction else float(all(passed)))
