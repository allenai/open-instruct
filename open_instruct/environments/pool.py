"""Pool of Ray actors with acquire/release semantics."""

import asyncio
from typing import Any

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@ray.remote
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is an async Ray actor. acquire() blocks until an actor is available
    (no polling needed — release() wakes up waiting acquirers via asyncio.Queue).
    """

    def __init__(self, pool_size: int, actor_class: type, **actor_kwargs: Any):
        remote_class = ray.remote(actor_class)

        logger.info(f"Creating pool of {pool_size} {actor_class.__name__} actors")
        self._actors = [remote_class.remote(**actor_kwargs) for _ in range(pool_size)]

        setup_tasks = [actor.setup.remote() for actor in self._actors]
        try:
            ray.get(setup_tasks)
        except Exception as e:
            logger.warning(f"Error during actor setup: {e}")

        self._available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)
        logger.info(f"Pool ready: {pool_size} {actor_class.__name__} actors")

    async def acquire(self) -> ray.actor.ActorHandle:
        """Block until an actor is available, then return it."""
        return await self._available.get()

    async def release(self, actor: ray.actor.ActorHandle) -> None:
        """Return an actor to the pool, waking any waiting acquirers."""
        await self._available.put(actor)

    def size(self) -> int:
        return len(self._actors)

    def available_count(self) -> int:
        return self._available.qsize()
