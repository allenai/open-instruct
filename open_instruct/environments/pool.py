"""Pool of Ray environment actors with acquire/release semantics."""

import asyncio
import logging
from typing import Any

import ray

from .base import RLEnvironment, get_env_class

logger = logging.getLogger(__name__)


class EnvironmentPool:
    """Pool of RLEnvironment Ray actors for concurrent rollouts."""

    def __init__(
        self,
        env_class: type[RLEnvironment],
        actors: list[ray.actor.ActorHandle],
        available: asyncio.Queue[ray.actor.ActorHandle],
    ):
        self._env_class = env_class
        self._actors = actors
        self._available = available

    @classmethod
    async def create(cls, pool_size: int, env_name: str, **env_kwargs: Any) -> "EnvironmentPool":
        """Create a pool of environment actors and call setup() on each."""
        env_class = get_env_class(env_name)
        actor_class = ray.remote(env_class)

        logger.info(f"Creating {pool_size} '{env_name}' environment actors")
        actors = [actor_class.remote(**env_kwargs) for _ in range(pool_size)]

        setup_tasks = [actor.setup.remote() for actor in actors]
        try:
            await asyncio.to_thread(ray.get, setup_tasks)
        except Exception as e:
            logger.warning(f"Error during environment setup: {e}")

        available: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for actor in actors:
            available.put_nowait(actor)

        logger.info(f"Environment pool initialized with {pool_size} '{env_name}' actors")
        return cls(env_class, actors, available)

    async def acquire(self) -> ray.actor.ActorHandle:
        """Acquire an available environment actor (blocks until available)."""
        return await self._available.get()

    def release(self, actor: ray.actor.ActorHandle) -> None:
        """Release an environment actor back to the pool."""
        self._available.put_nowait(actor)

    async def shutdown(self) -> None:
        """Call shutdown() on all actors and terminate them."""
        logger.info("Shutting down environment pool...")
        shutdown_tasks = [actor.shutdown.remote() for actor in self._actors]
        try:
            await asyncio.to_thread(ray.get, shutdown_tasks)
        except Exception as e:
            logger.warning(f"Error during environment shutdown: {e}")

        for actor in self._actors:
            ray.kill(actor)
        self._actors = []
        logger.info("Environment pool shutdown complete")

    @property
    def env_class(self) -> type[RLEnvironment]:
        return self._env_class

    def __len__(self) -> int:
        return len(self._actors)
