"""Preserve formatting predicates while avoiding repeated regex backtracking."""

import asyncio
import itertools
import json
import random
from types import SimpleNamespace

import pytest
import regex

from open_instruct.miles.rewards import candidate_verifiers as cv
from open_instruct.miles.rewards import rewards

QUOTED = {
    "kind": "format",
    "value": {"type": "regex", "pattern_id": "kv_quoted", "verify_regex": [r'".+":\s*".+"'], "verify_min_matches": 3},
}


def test_quoted_pair_matches_original_boolean_predicate():
    original = regex.compile(QUOTED["value"]["verify_regex"][0])
    # Exhaust short strings and vary whitespace/Unicode and embedded quotes.
    for n in range(9):
        for chars in itertools.product('":a ', repeat=n):
            line = "".join(chars)
            assert cv._quoted_pair_line(line) == bool(original.search(line)), repr(line)
    rng = random.Random(17)
    for _ in range(10000):
        line = "".join(rng.choices('":ab \t\r\u00a0\u001c☃', k=rng.randrange(100)))
        assert cv._quoted_pair_line(line) == bool(original.search(line)), repr(line)


@pytest.mark.parametrize("suffix", [': "' + "a" * 16000, ": " + "a" * 16000])
def test_quoted_pair_long_malformed_line_does_not_timeout(suffix):
    # Both cases hit the old 100ms timeout: every initial quote retried the suffix.
    malformed = '"' * 16000 + suffix
    assert cv.score(malformed, QUOTED) == 0
    valid = "\n".join(['"a": "b"', '"c": "d"', '"e": "f"'])
    assert cv.score(malformed + "\n" + valid, QUOTED) == 1
    assert cv.score(valid + "\n" + malformed, QUOTED) == 1


def test_line_minimum_short_circuits_only_after_enough_lines(monkeypatch):
    calls = []

    def search(pattern, line, **kwargs):
        calls.append(line)
        assert line != "unneeded tail"
        return line == "match"

    monkeypatch.setattr(cv.regex, "search", search)
    target = {"kind": "format", "value": {"type": "regex", "verify_regex": ["match"], "verify_min_matches": 2}}
    assert cv.score("match\nmiss\nmatch\nunneeded tail", target) == 1
    assert calls == ["match", "miss", "match"]
    assert cv.score("match", target) == 0


def test_unresolved_timeout_is_not_silently_zero(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.setattr(cv.regex, "search", timeout)
    target = {"kind": "format", "value": {"type": "regex", "pattern_id": "other", "verify_regex": ["x"]}}
    with pytest.raises(TimeoutError, match="pattern_id='other', response_chars=3"):
        cv.score("xxx", target)


def test_failed_verifier_input_is_saved_before_reraising(tmp_path, monkeypatch):
    class FailingVerifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            label["value"]["pattern_id"] = "mutated"
            raise TimeoutError("format timeout")

    monkeypatch.setattr(rewards, "_registry", lambda path: {"format": FailingVerifier()})
    args = SimpleNamespace(
        save=str(tmp_path),
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    response = "<think>reasoning</think>" + '"' * 32000
    sample = SimpleNamespace(
        metadata={"verifiers": [{"name": "format", "target": QUOTED}]},
        response=response,
        prompt="format this answer",
        status="completed",
        tokens=[1],
        response_length=1,
        group_index=7,
        index=9,
    )
    for _ in range(2):
        with pytest.raises(TimeoutError, match="format timeout"):
            asyncio.run(rewards.registered_reward(args, sample))
    paths = list((tmp_path / "verifier_failures").glob("*.json"))
    assert len(paths) == 1  # Bound failure capture even if other in-flight samples fail.
    record = json.loads(paths[0].read_text())
    assert record["response"] == response
    assert record["scored_response"] == response
    assert record["verifier"]["target"] == QUOTED
    assert record["group_index"] == 7
    assert "reward_components" not in sample.metadata


def test_failure_capture_io_error_does_not_hide_original_error(tmp_path, monkeypatch):
    class FailingVerifier:
        async def async_call(self, *args, **kwargs):
            raise TimeoutError("original error")

    monkeypatch.setattr(rewards, "_registry", lambda path: {"format": FailingVerifier()})
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    args = SimpleNamespace(
        save=str(blocked),
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    sample = SimpleNamespace(
        metadata={"verifiers": [{"name": "format", "target": QUOTED}]},
        response="unfinished",
        prompt="question",
        tokens=[],
        response_length=0,
    )
    with pytest.raises(TimeoutError, match="original error"):
        asyncio.run(rewards.registered_reward(args, sample))
