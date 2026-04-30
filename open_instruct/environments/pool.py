"""Pool of Ray actors with acquire/release semantics."""

import asyncio
import os
from typing import Any

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DEFAULT_ACQUIRE_TIMEOUT_S = 7200
ACQUIRE_CONCURRENCY = 1000
RELEASE_CONCURRENCY = 128


def _podman_docker_hosts_from_env() -> list[str]:
    hosts = os.getenv("SWERL_PODMAN_DOCKER_HOSTS", "")
    return [host.strip() for host in hosts.split(",") if host.strip()]


def _actor_kwargs_for_slot(actor_kwargs: dict[str, Any], slot_index: int, docker_hosts: list[str]) -> dict[str, Any]:
    kwargs = dict(actor_kwargs)
    if docker_hosts and kwargs.get("backend") == "docker" and "docker_host" not in kwargs:
        kwargs["docker_host"] = docker_hosts[slot_index % len(docker_hosts)]
    return kwargs


@ray.remote(concurrency_groups={"acquire": ACQUIRE_CONCURRENCY, "release": RELEASE_CONCURRENCY})
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is an async Ray actor. acquire() blocks until an actor is available
    (no polling needed — release() wakes up waiting acquirers via asyncio.Queue).
    """

    def __init__(
        self,
        pool_size: int,
        actor_class: type,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_S,
        **actor_kwargs: Any,
    ):
        self._acquire_timeout = acquire_timeout
        remote_class = ray.remote(actor_class)
        docker_hosts = _podman_docker_hosts_from_env()

        logger.info(f"Creating pool of {pool_size} {actor_class.__name__} actors")
        if docker_hosts:
            logger.info(
                "Assigning %s %s actors across %s Podman Docker hosts",
                pool_size,
                actor_class.__name__,
                len(docker_hosts),
            )
        self._actors = [
            remote_class.remote(**_actor_kwargs_for_slot(actor_kwargs, slot_index, docker_hosts))
            for slot_index in range(pool_size)
        ]

        setup_tasks = [actor.setup.remote() for actor in self._actors]
        ray.get(setup_tasks)

        self._available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)
        logger.info(f"Pool ready: {pool_size} {actor_class.__name__} actors")

    @ray.method(concurrency_group="acquire")
    async def acquire(self) -> ray.actor.ActorHandle:
        try:
            return await asyncio.wait_for(self._available.get(), timeout=self._acquire_timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Pool acquire timed out after {self._acquire_timeout}s. "
                f"Pool has {len(self._actors)} actors, {self._available.qsize()} available. "
                f"An actor may have crashed without being released."
            ) from e

    @ray.method(concurrency_group="release")
    async def release(self, actor: ray.actor.ActorHandle) -> None:
        await self._available.put(actor)

    def size(self) -> int:
        return len(self._actors)
