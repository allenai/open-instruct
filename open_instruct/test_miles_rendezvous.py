"""Staggered allocation retries must not consume previous-attempt readiness."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import rendezvous


def test_staggered_retry_waits_for_every_new_replica(tmp_path):
    assert rendezvous.poll_round(tmp_path, 0, 2, "a0") is None
    first = rendezvous.poll_round(tmp_path, 1, 2, "a1")
    assert rendezvous.poll_round(tmp_path, 0, 2, "a0") == first
    old = tmp_path / first
    old.mkdir()
    for name in ["complete.json", "failed-0.json", "head.json", "node-1.json"]:
        (old / name).write_text("old attempt")
    assert rendezvous.poll_round(tmp_path, 1, 2, "b1") is None
    assert rendezvous.poll_round(tmp_path, 1, 2, "b1") is None
    second = rendezvous.poll_round(tmp_path, 0, 2, "b0")
    assert second != first
    assert rendezvous.poll_round(tmp_path, 1, 2, "b1") == second
    assert not (tmp_path / second).exists()
    assert rendezvous.poll_round(tmp_path, 0, 2, "c0") is None
    third = rendezvous.poll_round(tmp_path, 1, 2, "c1")
    assert third not in [first, second]
    assert rendezvous.poll_round(tmp_path, 0, 2, "c0") == third


def test_single_replica_also_gets_new_attempt(tmp_path):
    assert rendezvous.join(tmp_path, 0, 1, 1) != rendezvous.join(tmp_path, 0, 1, 1)


def test_missing_restarted_peer_times_out(tmp_path):
    rendezvous.poll_round(tmp_path, 0, 2, "a0")
    rendezvous.poll_round(tmp_path, 1, 2, "a1")
    with pytest.raises(TimeoutError, match="fresh processes"):
        rendezvous.join(tmp_path, 0, 2, 0)


def test_topology_change_rejected(tmp_path):
    rendezvous.poll_round(tmp_path, 0, 2, "a0")
    with pytest.raises(InputError, match="Replica count changed"):
        rendezvous.poll_round(tmp_path, 0, 3, "b0")


@pytest.mark.parametrize("count", [2, 4])
def test_concurrent_replicas_agree_across_repeated_attempts(tmp_path, count):
    previous = set()
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=count) as pool:
            paths = list(pool.map(lambda rank: rendezvous.join(tmp_path, rank, count, 10), range(count)))
        assert len(set(paths)) == 1
        assert paths[0] not in previous
        previous.add(paths[0])
