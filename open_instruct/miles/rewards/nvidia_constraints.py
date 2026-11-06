"""Portable calendar and NeMo-target adapters using maintained Open Instruct IF checkers."""

import json
import re
import time
from functools import lru_cache

from open_instruct import ifeval_constraints, logger_utils
from open_instruct.IFEvalG import instructions_registry
from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import rewards, task_verifiers

logger = logger_utils.setup_logger(__name__)

# Native generators do not consistently validate types before storing arguments.
# Validate their count/index parameters before invoking them; equality with the
# returned arguments alone cannot distinguish a malformed but unchanged target.
_INTEGER_ARGUMENTS = {
    "N",
    "capital_frequency",
    "frequency",
    "let_frequency",
    "m",
    "n",
    "n_end",
    "n_sent",
    "n_start",
    "n_words",
    "nth_paragraph",
    "num_bullets",
    "num_highlights",
    "num_paragraphs",
    "num_placeholders",
    "num_sections",
    "num_sentences",
    "num_words",
    "small_n",
}
_LIST_ARGUMENTS = {"keywords", "forbidden_words"}
# These parameters still use `if not value` in their native generators.
# Copy indices use explicit None defaults, so zero is now a valid fixed index.
_RANDOM_ZERO_ARGUMENTS = {"N", "small_n", "n_sent", "n_words", "n", "m"}
_STRIPPED_ARGUMENTS = {
    "detectable_format:multiple_sections": ("section_spliter",),
    "detectable_content:postscript": ("postscript_marker",),
    "startend:end_checker": ("end_phrase",),
    "keywords:frequency": ("keyword",),
    "keywords:word_once": ("keyword",),
    "keywords:word_count_different_numbers": ("keyword",),
    "keywords:exclude_word_harder": ("keyword",),
    "keywords:keyword_specific_position": ("keyword",),
    "count:count_increment_word": ("keyword1", "keyword2"),
    "copy:repeat_phrase": ("phrase",),
    "first_word:first_word_sent": ("first_word",),
    "first_word:first_word_answer": ("first_word",),
    "last_word:last_word_sent": ("last_word",),
    "last_word:last_word_answer": ("last_word",),
}


def expected_native_args(name, args):
    """Allow only the normalization performed by this native constructor."""
    expected = dict(args)
    for key in _STRIPPED_ARGUMENTS.get(name, ()):
        expected[key] = args[key].strip()
    if name == "length_constraints:nth_paragraph_first_word":
        expected["first_word"] = args["first_word"].lower()
    elif name in {"keywords:letter_frequency", "letters:letter_counting2"}:
        # The native checker checks length before stripping and randomly replaces
        # invalid letters. Do not let a coincidental random match pass admission.
        if len(args["letter"]) != 1 or re.fullmatch("[a-z]", args["letter"].lower()) is None:
            raise ValueError(f"Invalid letter for {name} triggers a native random default")
        expected["letter"] = args["letter"].lower()
    elif name == "keywords:existence":
        expected["keywords"] = sorted(args["keywords"])
    elif name == "keywords:forbidden_words":
        expected["forbidden_words"] = sorted(set(args["forbidden_words"]))
    return expected


def validate_patterns(name, args):
    """Compile the patterns exactly as the native checkers use them."""
    patterns, flags = [], 0
    if name == "keywords:existence":
        patterns, flags = args["keywords"], re.IGNORECASE
    elif name == "keywords:forbidden_words":
        patterns, flags = [r"\b" + word + r"\b" for word in args["forbidden_words"]], re.IGNORECASE
    elif name in {"keywords:frequency", "keywords:word_once", "keywords:word_count_different_numbers"}:
        patterns, flags = [args["keyword"]], re.IGNORECASE
    elif name == "count:count_increment_word":
        patterns, flags = [args["keyword1"], args["keyword2"]], re.IGNORECASE
    elif name == "copy:repeat_phrase":
        words = args["phrase"].split()
        patterns = [rf"{words[0]} .*? {words[-1]}"]
    elif name == "detectable_format:multiple_sections":
        patterns = [r"\s?" + args["section_spliter"] + r"\s?\d+\s?"]
    elif name == "detectable_content:postscript" and args["postscript_marker"] not in {"P.P.S", "P.S."}:
        patterns, flags = [r"\s*" + args["postscript_marker"].lower() + r".*$"], re.MULTILINE
    try:
        for pattern in patterns:
            re.compile(pattern, flags=flags)
    except (re.error, RecursionError) as error:
        raise ValueError(f"Invalid native regex for {name}: {error}") from error


def minutes(value):
    if not isinstance(value, str):
        raise ValueError("Time must be a string")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", value.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError("Unsupported time")
    hour, minute, period = int(match[1]), int(match[2] or 0), (match[3] or "").lower()
    if minute >= 60 or (period and not 1 <= hour <= 12) or (not period and hour >= 24):
        raise ValueError("Time out of range")
    return ((hour % 12) + (12 if period == "pm" else 0) if period else hour) * 60 + minute


def calendar_contract(target):
    if not isinstance(target, dict):
        raise ValueError("Calendar target must be an object")
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


class _IsolatedAdapter:
    kind: str
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    def result(self, score, status="completed", **diagnostics):
        return common.Result(score, diagnostics={"kind": self.kind, "status": status, **diagnostics})

    async def async_call(self, tokenized_prediction, prediction, label, **kwargs):
        start = time.monotonic()
        try:
            return await rewards.isolated_verifier_call(
                {"factory": f"{type(self).__module__}.{type(self).__name__}"},
                tokenized_prediction,
                prediction,
                label,
                **kwargs,
            )
        except rewards.MathVerifierTimeout as error:
            logger.warning("%s verifier timeout; assigning zero reward: %s", self.kind, error)
            return self.result(
                0.0,
                status="timeout",
                error=str(error),
                failure_policy="zero",
                elapsed_seconds=time.monotonic() - start,
            )


class CalendarVerifier(_IsolatedAdapter):
    kind = "calendar_v1"

    def __call__(self, tokenized_prediction, prediction, label, **kwargs):
        contract = calendar_contract(label)
        try:
            events = common.strict_json(common.answer_text(prediction))
        except (ValueError, RecursionError) as error:
            return self.result(
                0.0, status="parse_error", error_type=type(error).__name__, error=str(error), failure_policy="zero"
            )
        try:
            if not isinstance(events, list) or len(events) != len(contract):
                return self.result(0.0)
            intervals, seen = [], set()
            for event in events:
                key = str(event["event_id"])
                if key not in contract or key in seen:
                    return self.result(0.0)
                seen.add(key)
                lower, upper, duration = contract[key]
                if type(event["duration"]) not in (int, float) or event["duration"] != duration:
                    return self.result(0.0)
                start = minutes(event["start_time"])
                if not lower <= start <= upper:
                    return self.result(0.0)
                intervals.append((start, start + duration))
            intervals.sort()
            valid = all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
            return self.result(float(valid))
        except (ValueError, TypeError, KeyError):
            return self.result(0.0)


@lru_cache(maxsize=4096)
def instruction_contract(serialized):
    label = json.loads(serialized)
    family, target = label["family"], label["target"]
    if family != "nemo":
        raise ValueError("Only the prepared NeMo instruction target schema is supported")
    registry = instructions_registry.INSTRUCTION_DICT
    ids = target["instruction_id_list"]
    arguments = target["kwargs"]
    issues = ifeval_constraints.if_training_issues({"instruction_id": ids, "kwargs": arguments})
    if issues:
        raise ValueError(f"Unsuitable IF training target: {issues}")
    grading_mode = target.get("grading_mode", "all")
    if grading_mode not in {"all", "fraction"}:
        raise ValueError("IF grading_mode must be all or fraction")
    compiled = []
    for name, args in zip(ids, arguments, strict=True):
        if name in {"language:response_language", "change_case:english_capital", "change_case:english_lowercase"}:
            raise ValueError("Language classification needs separate calibration")
        if name not in registry:
            raise ValueError(f"Unknown instruction family: {name}")
        checker = registry[name](name)
        args = {k: v for k, v in (args or {}).items() if v is not None}
        if set(args) != set(checker.get_instruction_args_keys()):
            raise ValueError(f"Missing or unexpected arguments for {name}")
        for key, value in args.items():
            expected_type = int if key in _INTEGER_ARGUMENTS else list if key in _LIST_ARGUMENTS else str
            if type(value) is not expected_type:
                raise ValueError(f"Argument {key} for {name} must be {expected_type.__name__}")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Empty argument {key}")
            if isinstance(value, list) and (not value or not all(isinstance(v, str) and v.strip() for v in value)):
                raise ValueError(f"Malformed argument {key}")
            if isinstance(value, int) and (value < 0):
                raise ValueError(f"Negative argument {key}; qualify separately")
            if key in _RANDOM_ZERO_ARGUMENTS and value == 0:
                raise ValueError(f"Zero argument {key} for {name} triggers a native random default")
        if name == "new:copy_span_idx" and not 0 <= args["n_start"] <= args["n_end"] < len(args["prompt_to_repeat"]):
            raise ValueError("Copy span must have ordered inclusive indices within the source text")
        if name == "copy:repeat_phrase" and len(args["phrase"].split()) < 3:
            raise ValueError("copy:repeat_phrase needs an interior word to replace")
        expected_args = expected_native_args(name, args)
        validate_patterns(name, expected_args)
        try:
            checker.build_description(**args)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"Invalid native checker arguments for {name}") from exc
        # Known normalization preserves native grading; other changes (including
        # random repairs) must not silently alter a training task.
        if (checker.get_instruction_args() or {}) != expected_args:
            raise ValueError(f"Native checker changed target arguments for {name}")
        compiled.append((name, checker, args))
    return grading_mode, compiled


class InstructionVerifier(_IsolatedAdapter):
    kind = "nemo_if_v1"

    def __call__(self, tokenized_prediction, prediction, label, **kwargs):
        grading_mode, compiled = instruction_contract(json.dumps(label, sort_keys=True, allow_nan=False))
        final = common.answer_text(prediction)
        if not final:
            return self.result(0.0)
        passed = []
        for name, checker, _args in compiled:
            value = checker.check_following(final)
            if type(value) is not bool:
                raise ValueError(f"IF checker {name!r} returned {type(value).__name__}, expected bool")
            passed.append(value)
        return self.result(sum(passed) / len(passed) if grading_mode == "fraction" else float(all(passed)))
