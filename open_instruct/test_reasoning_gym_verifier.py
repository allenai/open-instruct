"""Regression controls using real entries and scorers from the pinned optional library."""

import asyncio
import copy
import json
from importlib import metadata

import pytest

from open_instruct.miles.rewards import reasoning_gym as rg

library = pytest.importorskip("reasoning_gym")
pytestmark = pytest.mark.skipif(metadata.version("reasoning-gym") != "0.1.25", reason="Requires pinned reasoning-gym")


def test_default_scorer_family_coverage():
    verifier = rg.ReasoningGymVerifier()
    actual = {task for task in rg.TASKS if library.get_score_answer_fn(task).__func__ is verifier.default_scorer}
    assert actual == rg.DEFAULT_EXACT_TASKS


@pytest.mark.parametrize("task", ["base_conversion", "leg_counting", "simple_equations", "count_bits", "gcd"])
def test_default_scorers_require_exact_answers_on_real_entries(task):
    verifier = rg.ReasoningGymVerifier()
    dataset = library.create_dataset(task, seed=17, size=3)
    for entry in dataset:
        gold = entry["answer"]
        wrong = ["1" + gold, gold + "1", f"The answer is {gold}", "unrelated", ""]
        assert dataset.score_answer(answer=wrong[0], entry=entry) > 0  # Reproduce the old exploit.
        for answer in wrong:
            result = verifier.score(answer, entry)
            assert result.score == 0, (task, gold, answer)
            assert result.diagnostics["scoring_protocol"] == "exact_default_v2"
        for answer in [
            gold,
            f"  {gold}  ",
            f"<answer>{gold}</answer>",
            f"\\boxed{{{gold}}}",
            f"<answer> \\boxed{{ {gold} }} </answer>",
        ]:
            assert verifier.score(answer, entry).score == 1
        for answer in [
            f"<answer>wrong</answer><answer>{gold}</answer>",
            f"\\boxed{{wrong}} \\boxed{{{gold}}}",
            "<answer>\\boxed{wrong}</answer>",
            f"<answer>wrong</answer> \\boxed{{{gold}}}",
            f"<answer>\\boxed{{{gold}}}</answer> \\boxed{{wrong}}",
        ]:
            assert verifier.score(answer, entry).score == 0


@pytest.mark.parametrize("task", ["chain_sum", "word_sorting", "family_relationships", "needle_haystack"])
def test_custom_scorers_preserve_their_semantics(task):
    verifier = rg.ReasoningGymVerifier()
    dataset = library.create_dataset(task, seed=17, size=1)
    entry = dataset[0]
    for answer in [entry["answer"], entry["answer"] + " wrong", "unrelated", ""]:
        expected = dataset.score_answer(answer=answer, entry=entry)
        assert verifier.score(answer, entry).score == expected


@pytest.mark.parametrize("task", ["arc_agi", "rearc"])
def test_arc_parity_after_json_round_trip(task):
    dataset = library.create_dataset(task, seed=17, size=1)
    raw = dataset[0]
    prepared = json.loads(json.dumps(raw))
    before = copy.deepcopy(prepared)
    verifier = rg.ReasoningGymVerifier()
    # JSON arrays must be converted back to tuples for the library's grid equality.
    assert isinstance(raw["metadata"]["output"], tuple)
    assert dataset.score_answer(answer=raw["answer"], entry=raw) == 1.0
    assert dataset.score_answer(answer=raw["answer"], entry=prepared) != 1.0
    for answer in [raw["answer"], "0", "not a grid", ""]:
        expected = dataset.score_answer(answer=answer, entry=raw) if answer else 0.0
        assert verifier.score(answer, prepared).score == expected
    assert prepared == before  # Coercion must not mutate the saved reference.
    with pytest.raises(ValueError, match="Invalid ReasoningGym grid target"):
        asyncio.run(verifier.async_call([], raw["answer"], raw))


@pytest.mark.parametrize(
    "task",
    [
        "binary_matrix",
        "n_queens",
        "spiral_matrix",
        "string_insertion",
        "simple_integration",
        "intermediate_integration",
        "polynomial_multiplication",
        "countdown",
        "puzzle24",
    ],
)
def test_executable_answers_are_rejected_through_async_call(task, tmp_path):
    dataset = library.create_dataset(task, seed=17, size=1)
    entry = json.loads(json.dumps(dataset[0]))
    marker = tmp_path / "executed"
    answer = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    result = asyncio.run(rg.ReasoningGymVerifier().async_call([], answer, entry))
    assert result.score == 0
    assert result.diagnostics["status"] == "parse_error"
    assert not marker.exists()


@pytest.mark.parametrize(
    "task,answer",
    [
        ("boxnet", "[1]"),
        ("boxnet", "null"),
        ("game_of_life", "{}"),
        ("prime_factorization", "not factors"),
        ("coin_flip", "1e309"),
    ],
)
def test_malformed_responses_do_not_abort_training(task, answer):
    entry = json.loads(json.dumps(library.create_dataset(task, seed=17, size=1)[0]))
    result = asyncio.run(rg.ReasoningGymVerifier().async_call([], answer, entry))
    assert result.score == 0
    assert result.diagnostics["status"] == "parse_error"


def test_puzzle24_requires_the_actual_numbers_and_exact_result():
    entry = {"answer": "6+6+6+6", "metadata": {"source_dataset": "puzzle24", "numbers": [6, 6, 6, 6]}}
    verifier = rg.ReasoningGymVerifier()
    assert verifier.score("6+6+6+6", entry).score == 1
    assert verifier.score("1*2*3*4", entry).score == 0
    entry["metadata"]["numbers"] = [10, 10, 9, 2]
    assert verifier.score("10+10+9/2", entry).score == 0


def test_state_based_target_does_not_require_a_reference():
    entry = json.loads(json.dumps(library.create_dataset("graph_color", seed=17, size=1)[0]))
    assert entry["answer"] is None
    gold = json.dumps(entry["metadata"]["possible_answer"])
    assert asyncio.run(rg.ReasoningGymVerifier().async_call([], gold, entry)).score == 1


def test_timeout_returns_zero_and_diagnostics():
    entry = {
        "answer": "x**2 + C",
        "metadata": {"source_dataset": "simple_integration", "variable": "x", "integrand": "2*x"},
    }
    verifier = rg.ReasoningGymVerifier(rg.ReasoningGymConfig(timeout_seconds=1e-8))
    result = asyncio.run(verifier.async_call([], entry["answer"], entry))
    assert result.score == 0
    assert result.diagnostics["status"] == "timeout"


def test_invalid_configuration_stays_fatal():
    entry = {"answer": "__import__('os').getpid()", "metadata": {"source_dataset": "polynomial_multiplication"}}
    with pytest.raises(ValueError, match="Unsupported expression syntax"):
        asyncio.run(rg.ReasoningGymVerifier().async_call([], "1", entry))


@pytest.mark.parametrize(
    "task,answer,error_type",
    [
        ("number_sorting", "[1,", SyntaxError),
        ("coin_flip", "1/0", ZeroDivisionError),
        ("coin_flip", "1e310", OverflowError),
    ],
)
def test_invalid_references_remain_explicit_errors_through_async_call(task, answer, error_type, monkeypatch):
    entry = {"answer": answer, "metadata": {"source_dataset": task}}

    async def unexpected_scoring(*args, **kwargs):
        pytest.fail("A malformed target must not reach response scoring")

    monkeypatch.setattr(rg.rewards, "isolated_verifier_call", unexpected_scoring)
    with pytest.raises(ValueError, match=f"Invalid ReasoningGym reference for {task}") as error:
        asyncio.run(rg.ReasoningGymVerifier().async_call([], "1", entry))
    assert isinstance(error.value.__cause__, error_type)
