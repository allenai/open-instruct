"""Native instruction semantics, target admission, and isolated calendar rewards."""

import asyncio
import json
import random
from types import SimpleNamespace
from unittest import mock

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
    monkeypatch.setattr(nv, "instruction_contract", lambda _: ("all", [("unexpected:checker", checker, {})]))
    with pytest.raises(ValueError, match="unexpected:checker.*NoneType"):
        score(nv.InstructionVerifier(), "answer", {"target": {}})


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
        ("count:count_increment_word", {"keyword1": ["plan"], "keyword2": ["let"]}, "plan let let", "plan let"),
        ("detectable_format:json_format", {}, '{"a":1}', "hello"),
    ],
)
def test_native_controls(name, args, gold, wrong):
    label = target(name, args)
    verifier = nv.InstructionVerifier()
    assert verifier([], gold, label).score == 1
    assert verifier([], wrong, label).score == 0
    assert verifier([], "", label).score == 0
    assert verifier([], "thinking</think>" + gold, label).score == 1


def test_grading_mode_and_argument_repairs():
    verifier = nv.InstructionVerifier()
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
    verifier = nv.CalendarVerifier()
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
    assert asyncio.run(nv.InstructionVerifier().async_call([], "one two three", label)).score == 1
    assert asyncio.run(nv.CalendarVerifier().async_call([], "[]", {})).score == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("keywords:keyword_specific_position", {"keyword": "apple", "n": "1", "m": 1}),
        ("count:counting_composition", {"n_sent": "2", "n_words": 3}),
        ("copy:copying_multiple", {"prompt_to_repeat": "apple", "N": ["2"]}),
        ("copy:copy", {"prompt_to_repeat": 7}),
        ("keywords:existence", {"keywords": "apple"}),
        ("copy:repeat_phrase", {"phrase": "one two three", "small_n": True}),
    ],
)
def test_malformed_argument_types_are_fatal_before_scoring(name, args):
    label = target(name, args)
    with pytest.raises(ValueError, match="must be"):
        nv.instruction_contract(json.dumps(label))
    # Even an empty response must not mask a malformed target in the worker.
    with pytest.raises(RuntimeError, match="must be"):
        asyncio.run(nv.InstructionVerifier().async_call([], "", label))


def test_calendar_recursion_failure_through_worker():
    async def check():
        verifier = nv.CalendarVerifier()
        response = "[" * 10000 + "0" + "]" * 10000
        result = await verifier.async_call([], response, {})
        assert result.score == 0
        assert result.diagnostics["status"] == "parse_error"
        assert result.diagnostics["error_type"] == "RecursionError"
        assert result.diagnostics["failure_policy"] == "zero"
        assert (await verifier.async_call([], "[]", {})).score == 1
        with pytest.raises(RuntimeError, match="Calendar duration must be positive"):
            await verifier.async_call([], response, {"0": {"duration": 0}})

    asyncio.run(check())


@pytest.mark.parametrize("seed", [0, 2, 7, 17])
def test_zero_copy_span_is_supported_independently_of_rng(seed, monkeypatch):
    draw = mock.Mock(wraps=random.Random(seed).randint)
    monkeypatch.setattr(random, "randint", draw)
    nv.instruction_contract.cache_clear()
    args = {"prompt_to_repeat": "abcdef", "n_start": 0, "n_end": 4}
    verifier = nv.InstructionVerifier()
    assert verifier([], "abcde", target("new:copy_span_idx", args)).score == 1
    assert verifier([], "abcd", target("new:copy_span_idx", args)).score == 0
    args["n_start"] = 1
    assert verifier([], "bcde", target("new:copy_span_idx", args)).score == 1
    assert verifier([], "bcd", target("new:copy_span_idx", args)).score == 0
    draw.assert_not_called()


def test_zero_copy_span_in_worker():
    async def check():
        verifier = nv.InstructionVerifier()
        for start, end, gold, wrong in [(0, 4, "abcde", "abcd"), (0, 0, "a", "b")]:
            label = target("new:copy_span_idx", {"prompt_to_repeat": "abcdef", "n_start": start, "n_end": end})
            assert (await verifier.async_call([], gold, label)).score == 1
            assert (await verifier.async_call([], wrong, label)).score == 0

    asyncio.run(check())


@pytest.mark.parametrize("start,end", [(2, 1), (0, 6), (6, 6)])
def test_copy_span_invalid_bounds_are_fatal(start, end):
    label = target("new:copy_span_idx", {"prompt_to_repeat": "abcdef", "n_start": start, "n_end": end})
    with pytest.raises(ValueError, match="ordered inclusive indices"):
        nv.instruction_contract(json.dumps(label))


def test_explicit_zero_without_native_random_default_is_supported():
    label = target("detectable_format:number_bullet_lists", {"num_bullets": 0})
    assert nv.InstructionVerifier()([], "plain text", label).score == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("keywords:existence", {"keywords": ["["]}),
        ("keywords:word_once", {"keyword": "["}),
        ("keywords:forbidden_words", {"forbidden_words": ["["]}),
        ("keywords:frequency", {"keyword": "[", "frequency": 1, "relation": "at least"}),
        ("keywords:word_count_different_numbers", {"keyword": "[", "frequency": 1, "relation": "at least"}),
        ("count:count_increment_word", {"keyword1": "apple", "keyword2": "["}),
        ("copy:repeat_phrase", {"phrase": "[ two three", "small_n": 2}),
        ("detectable_format:multiple_sections", {"section_spliter": "[", "num_sections": 2}),
        ("detectable_content:postscript", {"postscript_marker": "["}),
    ],
)
def test_invalid_native_patterns_are_fatal_at_admission_and_in_worker(name, args):
    label = target(name, args)
    with pytest.raises(ValueError, match=f"Invalid native regex for {name}"):
        nv.instruction_contract(json.dumps(label))
    with pytest.raises(RuntimeError, match=f"Invalid native regex for {name}"):
        asyncio.run(nv.InstructionVerifier().async_call([], "", label))


def test_native_regex_semantics_are_preserved():
    verifier = nv.InstructionVerifier()
    label = target("keywords:existence", {"keywords": [r"appl[eE]+"]})
    assert verifier([], "APPLE", label).score == 1
    assert verifier([], "pear", label).score == 0
    label = target("keywords:forbidden_words", {"forbidden_words": ["apple"]})
    assert verifier([], "pineapple", label).score == 1
    assert verifier([], "APPLE", label).score == 0


@pytest.mark.parametrize("cls", [nv.CalendarVerifier, nv.InstructionVerifier])
def test_worker_timeout_scores_zero(cls, monkeypatch):
    worker = mock.AsyncMock(side_effect=nv.rewards.MathVerifierTimeout("worker deadline"))
    warning = mock.Mock()
    monkeypatch.setattr(nv.rewards, "isolated_verifier_call", worker)
    monkeypatch.setattr(nv.logger, "warning", warning)
    verifier = cls()
    result = asyncio.run(verifier.async_call([], "answer", {}))
    assert result.score == 0
    assert result.diagnostics["kind"] == verifier.kind
    assert result.diagnostics["status"] == "timeout"
    assert result.diagnostics["failure_policy"] == "zero"
    assert result.diagnostics["elapsed_seconds"] >= 0
    warning.assert_called_once()


@pytest.mark.parametrize("cls", [nv.CalendarVerifier, nv.InstructionVerifier])
def test_unexpected_worker_errors_remain_fatal(cls, monkeypatch):
    monkeypatch.setattr(nv.rewards, "isolated_verifier_call", mock.AsyncMock(side_effect=RuntimeError("bad target")))
    with pytest.raises(RuntimeError, match="bad target"):
        asyncio.run(cls().async_call([], "answer", {}))


def test_real_regex_timeout_and_worker_recovery():
    async def check():
        loop = asyncio.get_running_loop()
        pool = nv.rewards._MathProcessPool(workers=1, timeout=5)
        nv.rewards._POOLS[loop] = pool
        verifier = nv.InstructionVerifier()
        label = target("keywords:existence", {"keywords": ["^(a+)+$"]})
        try:
            # Warm the worker before reducing its deadline to bound regex work.
            assert (await verifier.async_call([], "a", label)).score == 1
            pool.timeout = 0.1
            call = verifier.async_call([], "a" * 80 + "!", label)
            result = await call
            assert result.score == 0
            assert result.diagnostics["kind"] == "nemo_if_v1"
            assert result.diagnostics["status"] == "timeout"
            pool.timeout = 5
            assert (await verifier.async_call([], "a", label)).score == 1
        finally:
            await pool.close()
            nv.rewards._POOLS.pop(loop, None)

    asyncio.run(check())


def test_calendar_time_case_and_raw_json_contract():
    async def check():
        verifier = nv.CalendarVerifier()
        label = {"0": {"duration": 60, "min_time": "9AM", "max_time": "5PM", "constraint": "before 11am"}}
        response = '[{"event_id": 0, "duration": 60, "start_time": "9:00 AM"}]'
        result = await verifier.async_call([], response, label)
        assert result.score == 1
        assert result.diagnostics == {"kind": "calendar_v1", "status": "completed"}
        result = await verifier.async_call([], response.replace("9:00 AM", "9:00 PM"), label)
        assert result.score == 0
        assert result.diagnostics == {"kind": "calendar_v1", "status": "completed"}
        result = await verifier.async_call([], "```json\n" + response + "\n```", label)
        assert result.score == 0
        assert result.diagnostics["status"] == "parse_error"
        assert result.diagnostics["kind"] == "calendar_v1"

    asyncio.run(check())


def test_instruction_completed_diagnostics_through_worker():
    async def check():
        verifier = nv.InstructionVerifier()
        for text, expected in [("apple", 1), ("pear", 0), ("", 0)]:
            result = await verifier.async_call([], text, target("keywords:existence", {"keywords": ["apple"]}))
            assert result.score == expected
            assert result.diagnostics == {"kind": "nemo_if_v1", "status": "completed"}

    asyncio.run(check())


@pytest.mark.parametrize(
    "name,args,normalized,gold,wrong",
    [
        (
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 1, "nth_paragraph": 1, "first_word": "The"},
            {"first_word": "the"},
            "THE cat sleeps.",
            "A cat sleeps.",
        ),
        *[
            (name, {"letter": "A", "let_frequency": 2, "let_relation": "at least"}, {"letter": "a"}, "aA", "b")
            for name in ["keywords:letter_frequency", "letters:letter_counting2"]
        ],
        (
            "keywords:frequency",
            {"keyword": " apple ", "frequency": 1, "relation": "at least"},
            {"keyword": "apple"},
            "apple",
            "pear",
        ),
        ("keywords:word_once", {"keyword": " apple "}, {"keyword": "apple"}, "apple", "pear"),
        (
            "keywords:word_count_different_numbers",
            {"keyword": " apple ", "frequency": 1, "relation": "at least"},
            {"keyword": "apple"},
            "apple",
            "pear",
        ),
        (
            "keywords:exclude_word_harder",
            {"keyword": " apple "},
            {"keyword": "apple"},
            "a pear today",
            "an apple today",
        ),
        (
            "keywords:keyword_specific_position",
            {"keyword": " apple ", "n": 1, "m": 1},
            {"keyword": "apple"},
            "apple today.",
            "pear today.",
        ),
        (
            "count:count_increment_word",
            {"keyword1": " apple ", "keyword2": " pear "},
            {"keyword1": "apple", "keyword2": "pear"},
            "apple pear pear",
            "apple pear",
        ),
        (
            "copy:repeat_phrase",
            {"phrase": " one two three ", "small_n": 1},
            {"phrase": "one two three"},
            "one four three",
            "one two three",
        ),
        *[
            (name, {"first_word": " The "}, {"first_word": "The"}, "The cat sleeps.", "A cat sleeps.")
            for name in ["first_word:first_word_sent", "first_word:first_word_answer"]
        ],
        *[
            (name, {"last_word": " cat "}, {"last_word": "cat"}, "A cat.", "A dog.")
            for name in ["last_word:last_word_sent", "last_word:last_word_answer"]
        ],
        (
            "detectable_format:multiple_sections",
            {"section_spliter": " Section ", "num_sections": 1},
            {"section_spliter": "Section"},
            "Section 1 content",
            "plain content",
        ),
        (
            "detectable_content:postscript",
            {"postscript_marker": " P.S. "},
            {"postscript_marker": "P.S."},
            "P.S. hello",
            "plain content",
        ),
        ("startend:end_checker", {"end_phrase": " Goodbye "}, {"end_phrase": "Goodbye"}, "Goodbye", "Hello"),
    ],
)
def test_native_normalization_preserves_rewards(name, args, normalized, gold, wrong):
    normalized = {**args, **normalized}
    checker = instructions_registry.INSTRUCTION_DICT[name](name)
    checker.build_description(**args)

    async def check():
        verifier = nv.InstructionVerifier()
        for response, expected in [(gold, 1), (wrong, 0)]:
            assert float(checker.check_following(response)) == expected
            original = await verifier.async_call([], response, target(name, args))
            canonical = await verifier.async_call([], response, target(name, normalized))
            assert original.score == canonical.score == expected

    asyncio.run(check())


def test_regex_validation_uses_native_stripped_pattern():
    label = target("keywords:word_once", {"keyword": " (?i)apple "})
    assert asyncio.run(nv.InstructionVerifier().async_call([], "APPLE", label)).score == 1
    # Valid before stripping, but the native checker would see a dangling escape.
    invalid = target("keywords:word_once", {"keyword": "apple\\ "})
    with pytest.raises(RuntimeError, match="Invalid native regex"):
        asyncio.run(nv.InstructionVerifier().async_call([], "apple", invalid))


@pytest.mark.parametrize("letter", [" A ", "aa", "1", "İ"])
def test_invalid_letters_cannot_be_randomly_repaired(letter, monkeypatch):
    choice = mock.Mock(return_value="a")
    monkeypatch.setattr(random, "choice", choice)
    nv.instruction_contract.cache_clear()
    label = target("keywords:letter_frequency", {"letter": letter, "let_frequency": 1, "let_relation": "at least"})
    with pytest.raises(ValueError, match="native random default"):
        nv.instruction_contract(json.dumps(label))
    choice.assert_not_called()


def test_normalization_does_not_allow_numeric_repairs(monkeypatch):
    nv.instruction_contract.cache_clear()
    monkeypatch.setattr(
        instructions_registry.INSTRUCTION_DICT["keywords:frequency"],
        "get_instruction_args",
        lambda self: {"keyword": "apple", "frequency": 2, "relation": "at least"},
    )
    label = target("keywords:frequency", {"keyword": " apple ", "frequency": 1, "relation": "at least"})
    with pytest.raises(ValueError, match="Native checker changed target arguments"):
        nv.instruction_contract(json.dumps(label))


def test_whitespace_is_not_normalized_for_unlisted_arguments():
    label = target("keywords:existence", {"keywords": [" apple "]})
    assert nv.InstructionVerifier()([], "apple", label).score == 0
    assert nv.InstructionVerifier()([], "an apple today", label).score == 1


def test_bigram_whole_answer_in_worker_and_fractional_mode():
    async def check():
        label = target("detectable_format:bigram_wrapping", {})
        verifier = nv.InstructionVerifier()
        for answer, expected in [
            ("<<hello world>>", 1),
            ("hello", 0),
            ("<<hello world>> stray", 0),
            ("<<hello world>> <<again>>", 1),
            ("<<hello>>", 1),
            ("<<hello>> <<world again>>", 0),
        ]:
            assert (await verifier.async_call([], answer, label)).score == expected
        label["target"]["instruction_id_list"].append("keywords:frequency")
        label["target"]["kwargs"].append({"keyword": "hello", "frequency": 1, "relation": "at least"})
        label["target"]["grading_mode"] = "fraction"
        assert (await verifier.async_call([], "hello hello", label)).score == 0.5
        label["target"]["grading_mode"] = "all"
        assert (await verifier.async_call([], "hello hello", label)).score == 0.0

    asyncio.run(check())


def test_legacy_nemo_keyword_stays_exactly_once_in_worker():
    async def check():
        label = target("keywords:word_once", {"keyword": "apple"})
        verifier = nv.InstructionVerifier()
        assert (await verifier.async_call([], "apple", label)).score == 1
        assert (await verifier.async_call([], "apple apple", label)).score == 0

    asyncio.run(check())
