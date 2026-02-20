"""Pool of Ray actors with acquire/release semantics."""

import asyncio
from typing import Any

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DEFAULT_ACQUIRE_TIMEOUT_S = 300


@ray.remote
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

        logger.info(f"Creating pool of {pool_size} {actor_class.__name__} actors")
        self._actors = [remote_class.remote(**actor_kwargs) for _ in range(pool_size)]

        setup_tasks = [actor.setup.remote() for actor in self._actors]
        ray.get(setup_tasks)

        self._available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)
        logger.info(f"Pool ready: {pool_size} {actor_class.__name__} actors")

    async def acquire(self) -> ray.actor.ActorHandle:
        try:
            return await asyncio.wait_for(self._available.get(), timeout=self._acquire_timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Pool acquire timed out after {self._acquire_timeout}s. "
                f"Pool has {len(self._actors)} actors, {self._available.qsize()} available. "
                f"An actor may have crashed without being released."
            ) from e

    async def release(self, actor: ray.actor.ActorHandle) -> None:
        await self._available.put(actor)

    def size(self) -> int:
        return len(self._actors)
