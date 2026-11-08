"""DocQA verifier controls for the cache mechanics image."""

import asyncio

import pytest

from open_instruct.miles.rewards import long_context


@pytest.mark.parametrize(
    "response,expected", [("The correct answer is C", 1), ("The correct answer is B", 0), ("A B C D", 0)]
)
def test_docqa_probe_contract(response, expected):
    target = {"version": 1, "metric": "choice", "answers": ["C"], "extraction": "choice"}
    result = asyncio.run(long_context.LongContextVerifier().async_call([], response, target))
    assert result.score == expected
