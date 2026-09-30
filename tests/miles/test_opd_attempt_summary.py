import pytest
from scripts.miles import summarize_opd_attempts


def event(attempt, name, prompt="p", lengths=(10, 20)):
    return {
        "attempt_id": attempt,
        "event": name,
        "prompt_sha256": prompt,
        "response_lengths": list(lengths),
        "statuses": ["completed", "completed"],
        "time_ns": 1000,
        "rollout_id": 0,
    }


def test_recycling_and_censoring_have_distinct_counting_units():
    rows = [event("a", "submitted"), event("a", "aborted_group_rejected")]
    rows += [event("b", name) for name in ("submitted", "generation_returned", "queue_selected", "batch_delivered")]
    rows += [event("c", "submitted", prompt="q")]
    result = summarize_opd_attempts.summarize(rows)
    assert result["attempts"] == 3
    assert result["unique_submitted_prompt_hashes"] == 2
    assert result["unique_delivered_prompt_hashes"] == 1
    assert result["submission_count_histogram_by_prompt"] == {1: 1, 2: 1}
    assert result["attempt_outcomes"] == {"rejected": 1, "delivered": 1, "unresolved_at_file_end": 1}
    assert result["delivered"]["groups"] == 1
    assert result["delivered"]["responses"] == 2
    assert result["completed_returns_not_delivered_as_of_file_end"]["groups"] == 0


def test_duplicate_delivery_is_not_silently_double_counted():
    rows = [event("a", "submitted"), event("a", "batch_delivered"), event("a", "batch_delivered")]
    with pytest.raises(ValueError, match="at most one"):
        summarize_opd_attempts.summarize(rows)
