"""Failure qualification rejects missing, unbounded or recovered admission evidence."""

import pytest
from scripts.miles.engine_drain_failure_probe import validate_failure


def test_failed_engine_stays_unavailable_while_peer_can_reopen():
    validate_failure(
        [
            {"event": "engine_unavailable", "engine": "1", "time": 12},
            {"event": "engine_reopened", "engine": "0", "time": 13},
        ],
        {"engine": "1", "time": 10},
        True,
        5,
    )


@pytest.mark.parametrize("kind", ["no_injection", "no_failure", "late", "historical", "wrong_engine", "reopened"])
def test_false_failure_evidence_is_rejected(kind):
    events = [{"event": "engine_unavailable", "engine": "1", "time": 12}]
    injected = {"engine": "1", "time": 10}
    failed, elapsed = True, 5
    if kind == "no_injection":
        injected = {}
    elif kind == "no_failure":
        failed = False
    elif kind == "late":
        elapsed = 601
    elif kind == "historical":
        events[0]["time"] = 9
    elif kind == "wrong_engine":
        events[0]["engine"] = "0"
    else:
        events.append({"event": "engine_reopened", "engine": "1", "time": 13})
    with pytest.raises(AssertionError):
        validate_failure(events, injected, failed, elapsed)
