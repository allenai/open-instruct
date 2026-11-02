"""Regression controls using real entries and scorers from the pinned optional library."""

import asyncio
from importlib import metadata

import pytest

from open_instruct.miles.rewards import reasoning_gym as rg

library = pytest.importorskip("reasoning_gym")
pytestmark = pytest.mark.skipif(metadata.version("reasoning-gym") != "0.1.25", reason="Requires pinned reasoning-gym")


def test_default_scorer_family_coverage():
    verifier = rg.ReasoningGymVerifier()
    actual = {task for task in rg.TASKS if library.get_score_answer_fn(task).__func__ is verifier.default_scorer}
    assert actual == rg.DEFAULT_EXACT_TASKS


@pytest.mark.parametrize("task", sorted(rg.DEFAULT_EXACT_TASKS))
def test_default_scorers_require_exact_answers_on_real_entries(task):
    verifier = rg.ReasoningGymVerifier()
    dataset = library.create_dataset(task, seed=17, size=3)
    for entry in dataset:
        gold = entry["answer"]
        wrong = ["1" + gold, gold + "1", f"The answer is {gold}", "unrelated", ""]
        assert dataset.score_answer(answer=wrong[0], entry=entry) > 0  # Reproduce the old exploit.
        for answer in wrong:
            result = asyncio.run(verifier.async_call([], answer, entry))
            assert result.score == 0, (task, gold, answer)
            assert result.diagnostics["scoring_protocol"] == "exact_default_v2"
        for answer in [gold, f"  {gold}  ", f"<answer>{gold}</answer>", f"\\boxed{{{gold}}}"]:
            assert asyncio.run(verifier.async_call([], answer, entry)).score == 1
        for answer in [f"<answer>wrong</answer><answer>{gold}</answer>", f"\\boxed{{wrong}} \\boxed{{{gold}}}"]:
            assert asyncio.run(verifier.async_call([], answer, entry)).score == 0


@pytest.mark.parametrize("task", ["chain_sum", "word_sorting", "family_relationships", "needle_haystack"])
def test_custom_scorers_preserve_their_semantics(task):
    verifier = rg.ReasoningGymVerifier()
    dataset = library.create_dataset(task, seed=17, size=1)
    entry = dataset[0]
    for answer in [entry["answer"], entry["answer"] + " wrong", "unrelated", ""]:
        expected = dataset.score_answer(answer=answer, entry=entry)
        assert asyncio.run(verifier.async_call([], answer, entry)).score == expected
