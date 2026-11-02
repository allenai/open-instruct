import asyncio
from types import SimpleNamespace

import pytest

from open_instruct.miles.rewards import candidate_code, candidate_verifiers


@pytest.mark.parametrize("text", ["B", "Answer: B", "Explanation.\n\\boxed{B}", "<think>A</think>Answer: B."])
def test_final_choice_positive(text):
    assert candidate_verifiers.score(text, {"kind": "mcqa_final_choice", "value": "B", "choices": ["A", "B"]}) == 1


@pytest.mark.parametrize(
    "text", ["A", "", "Answer: A\nAnswer: B", "\\boxed{A} \\boxed{B}", "B or A", "Answer: B\nActually A"]
)
def test_final_choice_shortcuts(text):
    assert candidate_verifiers.score(text, {"kind": "mcqa_final_choice", "value": "B", "choices": ["A", "B"]}) == 0


@pytest.mark.parametrize("returned", [[True], [True, "yes"], [True, 2]])
def test_code_incomplete_or_invalid_results_raise(monkeypatch, returned):
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"results": returned})
    monkeypatch.setattr(
        candidate_code.code_rewards, "_get_session", lambda: SimpleNamespace(post=lambda *a, **kw: response)
    )
    verifier = candidate_code.CompleteCodeVerifier(candidate_code.Config("http://unused"))
    with pytest.raises(RuntimeError):
        asyncio.run(verifier.async_call([], "def f(): return 1", ["assert f()==1", "assert f()==2"]))
