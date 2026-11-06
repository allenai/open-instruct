"""Preserve formatting predicates while avoiding repeated regex backtracking."""

import asyncio
import itertools
import json
import random
from types import SimpleNamespace

import pytest
import regex

from open_instruct.miles.rewards import deterministic_verifiers as cv
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


def test_quoted_pattern_variants_keep_regex_semantics(monkeypatch):
    # An anchored variant must reject a prefix accepted by the unanchored rule.
    line = 'prefix "a": "b"'
    assert QUOTED["value"]["verify_regex"][0] == cv._QUOTED_PAIR_PATTERN
    assert cv._quoted_pair_line(line)

    def unexpected_fast_path(line):
        pytest.fail("Pattern variants must use the regex fallback")

    monkeypatch.setattr(cv, "_quoted_pair_line", unexpected_fast_path)
    target = {"kind": "format", "value": {"type": "regex", "verify_regex": [r'^".+":\s*".+"$']}}
    assert cv.score(line, target) == 0
    assert cv.score('"a": "b"', target) == 1


def test_unresolved_timeout_is_not_silently_zero(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.setattr(cv.regex, "search", timeout)
    target = {"kind": "format", "value": {"type": "regex", "pattern_id": "other", "verify_regex": ["x"]}}
    with pytest.raises(
        TimeoutError, match="pattern_id='other', response_chars=3; original_error=regex timed out"
    ) as caught:
        cv.score("xxx", target)
    assert str(caught.value.__cause__) == "regex timed out"


def test_failed_verifier_input_is_saved_before_reraising(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    class FailingVerifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            label["value"]["pattern_id"] = "mutated"
            label["value"]["verify_regex"] = ["x"]
            cv.score(prediction, label)

    monkeypatch.setattr(cv.regex, "search", timeout)
    monkeypatch.setattr(rewards, "_FAILURE_CAPTURED", set())
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
        with pytest.raises(TimeoutError, match="original_error=regex timed out"):
            asyncio.run(rewards.registered_reward(args, sample))
    paths = list((tmp_path / "verifier_failures").glob("*.json"))
    assert len(paths) == 1  # Bound failure capture even if other in-flight samples fail.
    record = json.loads(paths[0].read_text())
    assert record["response"] == response
    assert record["scored_response"] == response
    assert record["verifier"]["target"] == QUOTED
    assert "pattern_id='mutated'" in record["error"]
    assert "original_error=regex timed out" in record["error"]
    assert record["group_index"] == 7
    assert "reward_components" not in sample.metadata


def test_failure_capture_io_error_does_not_hide_original_error(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_FAILURE_CAPTURED", set())

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


@pytest.mark.parametrize("kind", ["regex", "inline_prose", "string_match"])
@pytest.mark.parametrize("policy", ["zero", "raise"])
def test_real_general_pattern_timeout_policy(kind, policy, monkeypatch):
    monkeypatch.delenv("OI_MILES_FORMAT_TIMEOUT_POLICY", raising=False)
    monkeypatch.setenv("OI_MILES_MATH_TIMEOUT_POLICY", policy)
    value = {"type": kind, "pattern_id": "nested_repetition"}
    if kind == "string_match":
        value.update(expected_markers=["a"], patterns=["^(a|aa)+$"])
    else:
        value["verify_regex"] = ["^(a|aa)+$"]
    verifier = cv.DeterministicVerifier()
    call = verifier.async_call([], "a" * 10000 + "!", {"kind": "format", "value": value})
    if policy == "raise":
        with pytest.raises(cv.FormattingVerifierTimeout, match="nested_repetition"):
            asyncio.run(call)
    else:
        result = asyncio.run(call)
        assert result.score == 0.0
        assert result.diagnostics["status"] == "timeout"
        assert result.diagnostics["failure_policy"] == "zero"
        assert "regex timed out" in result.diagnostics["error"]
        assert result.diagnostics["elapsed_seconds"] > 0


def test_timeout_zero_preserves_input_and_diagnostics_in_reward_bridge(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.setattr(cv.regex, "search", timeout)
    monkeypatch.setattr(rewards, "_FAILURE_CAPTURED", set())
    monkeypatch.setattr(rewards, "_registry", lambda path: {"format": cv.DeterministicVerifier()})
    monkeypatch.setenv("OI_MILES_MATH_TIMEOUT_POLICY", "raise")
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "zero")  # Explicit format policy takes precedence.
    target = {"kind": "format", "value": {"type": "regex", "verify_regex": ["x"]}}
    args = SimpleNamespace(
        save=str(tmp_path),
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    sample = SimpleNamespace(
        metadata={"verifiers": [{"name": "format", "target": target}]},
        response="x" * 32000,
        prompt="format this answer",
        tokens=[],
        response_length=0,
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0.0
    assert sample.metadata["verifier_diagnostics"]["format"]["status"] == "timeout"
    record = json.loads(next((tmp_path / "verifier_failures").glob("*.json")).read_text())
    assert record["response"] == sample.response
    assert record["verifier"]["target"] == target
    assert "regex timed out" in record["error"]


def test_timeout_policy_rejects_invalid_policy_and_preserves_bad_targets(monkeypatch):
    verifier = cv.DeterministicVerifier()
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "invalid")
    with pytest.raises(ValueError, match="must be zero or raise"):
        asyncio.run(verifier.async_call([], "", QUOTED))
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "zero")
    with pytest.raises(ValueError, match="Invalid formatting patterns"):
        asyncio.run(verifier.async_call([], "", {"kind": "format", "value": {"type": "regex", "verify_regex": []}}))


def test_default_timeout_policy_is_zero(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.delenv("OI_MILES_FORMAT_TIMEOUT_POLICY", raising=False)
    monkeypatch.delenv("OI_MILES_MATH_TIMEOUT_POLICY", raising=False)
    monkeypatch.setattr(cv.regex, "search", timeout)
    target = {"kind": "format", "value": {"type": "regex", "verify_regex": ["x"]}}
    result = asyncio.run(cv.DeterministicVerifier().async_call([], "x", target))
    assert result.score == 0.0 and result.diagnostics["status"] == "timeout"


@pytest.mark.parametrize("diagnostics", [{"kind": "math", "status": "timeout"}, {"status": "timeout", "error": None}])
def test_returned_timeout_cannot_consume_exception_capture_slot(tmp_path, monkeypatch, diagnostics):
    class Verifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            if prediction == "timeout response":
                return cv.Result(0.0, diagnostics=diagnostics)
            raise ValueError("later fatal verifier error")

    monkeypatch.setattr(rewards, "_FAILURE_CAPTURED", set())
    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": Verifier()})
    args = SimpleNamespace(
        save=str(tmp_path),
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    sample = SimpleNamespace(
        metadata={"verifiers": [{"name": "test", "target": {}}]},
        response="timeout response",
        prompt="question",
        tokens=[],
        response_length=0,
    )
    for _ in range(2):
        assert asyncio.run(rewards.registered_reward(args, sample)) == 0.0
    sample.response = "fatal response"
    for _ in range(2):
        with pytest.raises(ValueError, match="later fatal verifier error"):
            asyncio.run(rewards.registered_reward(args, sample))
    records = [json.loads(path.read_text()) for path in (tmp_path / "verifier_failures").glob("*.json")]
    assert len(records) == 2  # One bounded slot for each purpose.
    by_kind = {record["capture_kind"]: record for record in records}
    assert by_kind["timeout"]["response"] == "timeout response"
    assert "without an error message" in by_kind["timeout"]["error"]
    assert by_kind["exception"]["response"] == "fatal response"
    assert "later fatal verifier error" in by_kind["exception"]["error"]


@pytest.mark.parametrize("mode", ["lines", "patterns", "quoted", "string_match"])
def test_response_deadline_bounds_many_individually_fast_checks(monkeypatch, mode):
    elapsed, budgets = [0.0], []
    monkeypatch.setattr(cv, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))

    def fast_check(*args, timeout):
        budgets.append(timeout)
        elapsed[0] += 0.04  # Each operation is below the individual 100ms cap.
        return None

    def fast_finditer(*args, timeout):
        fast_check(timeout=timeout)
        yield SimpleNamespace(group=lambda: "match")

    monkeypatch.setattr(cv.regex, "search", fast_check)
    monkeypatch.setattr(cv.regex, "finditer", fast_finditer)
    monkeypatch.setattr(cv, "_quoted_pair_line", fast_check)
    value = {"type": "regex", "verify_regex": ["miss"]}
    text = "\n".join(["line"] * 1000)
    if mode == "patterns":
        value["verify_regex"] *= 1000
        text = "one line"
    elif mode == "quoted":
        value["verify_regex"] = [cv._QUOTED_PAIR_PATTERN]
    elif mode == "string_match":
        value = {"type": "string_match", "expected_markers": ["match"], "patterns": ["match"] * 1000}
        text = "match"
    with pytest.raises(cv.FormattingVerifierTimeout, match="response matching deadline exceeded"):
        cv.score(text, {"kind": "format", "value": value})
    assert 1 < len(budgets) < 1000
    assert all(0 < budget <= cv._FORMAT_SEARCH_TIMEOUT_SECONDS for budget in budgets)
    assert budgets[-1] < budgets[0]  # Calls near the deadline get only the remaining budget.
    assert cv._FORMAT_RESPONSE_TIMEOUT_SECONDS <= elapsed[0] < cv._FORMAT_RESPONSE_TIMEOUT_SECONDS + 0.05
