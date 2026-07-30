import asyncio

import pytest

from open_instruct.environments.pool import EnvironmentPool, _is_podman_host_failure

# EnvironmentPool is a Ray actor class; these tests drive the undecorated
# implementation's async methods directly, so no Ray runtime is needed.
_PoolImpl = EnvironmentPool.__ray_metadata__.modified_class


def _make_pool(num_actors: int = 2):
    """Build a pool instance without running __init__ (which spawns Ray actors)."""
    pool = object.__new__(_PoolImpl)
    pool._acquire_timeout = 5.0
    pool._actors = [object() for _ in range(num_actors)]
    pool._available = asyncio.Queue()
    for actor in pool._actors:
        pool._available.put_nowait(actor)
    pool._docker_hosts = []
    pool._actor_host_leases = {}
    pool._waiting = 0
    pool._in_use_peak = 0
    pool._acquire_wait_sum_s = 0.0
    pool._acquire_count = 0
    return pool


def test_pool_stats_track_utilization():
    async def scenario():
        pool = _make_pool(num_actors=2)
        stats = pool.stats()
        assert stats["size"] == 2
        assert stats["in_use"] == 0
        assert stats["waiting"] == 0

        first = await pool._acquire_actor()
        await pool._acquire_actor()
        stats = pool.stats()
        assert stats["in_use"] == 2
        assert stats["in_use_peak"] == 2
        assert stats["acquires"] == 2

        # A third acquire must block; stats() taken meanwhile reports the waiter.
        blocked = asyncio.create_task(pool._acquire_actor())
        await asyncio.sleep(0.05)
        stats = pool.stats()
        assert stats["waiting"] == 1
        assert stats["in_use"] == 2

        await pool._release_actor(first)
        reacquired = await blocked
        assert reacquired is first
        stats = pool.stats()
        assert stats["in_use"] == 2
        assert stats["acquires"] == 1
        assert stats["acquire_wait_s_mean"] > 0.01  # the blocked acquire waited for the release

    asyncio.run(scenario())


def test_pool_stats_peak_resets_between_calls():
    async def scenario():
        pool = _make_pool(num_actors=2)
        first = await pool._acquire_actor()
        second = await pool._acquire_actor()
        assert pool.stats()["in_use_peak"] == 2

        await pool._release_actor(first)
        await pool._release_actor(second)
        # Peak was reset to the current in-use level by the previous stats().
        stats = pool.stats()
        assert stats["in_use"] == 0
        assert stats["in_use_peak"] == 2  # peak observed before the releases carried over
        assert pool.stats()["in_use_peak"] == 0

    asyncio.run(scenario())


def test_pool_acquire_timeout_leaves_counters_consistent():
    async def scenario():
        pool = _make_pool(num_actors=1)
        pool._acquire_timeout = 0.05
        await pool._acquire_actor()
        with pytest.raises(TimeoutError):
            await pool._acquire_actor()
        stats = pool.stats()
        assert stats["waiting"] == 0
        assert stats["in_use"] == 1

    asyncio.run(scenario())


def test_is_podman_host_failure_detects_unresponsive_socket():
    error = RuntimeError(
        "Reset failed after 5 attempts: Error while fetching server API version: "
        "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out. (read timeout=300)"
    )

    assert _is_podman_host_failure(error)


def test_is_podman_host_failure_detects_connection_errors():
    assert _is_podman_host_failure(ConnectionError("Connection refused while connecting to podman.sock"))


def test_is_podman_host_failure_ignores_task_failures():
    assert not _is_podman_host_failure(RuntimeError("Command timed out after 120s."))
