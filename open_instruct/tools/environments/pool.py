"""Pool of Ray environment actors with acquire/release semantics."""

import asyncio
import logging
from typing import Any

import ray

from .base import RLEnvironment, get_env_class

logger = logging.getLogger(__name__)


class EnvironmentPool:
    """Pool of RLEnvironment Ray actors for concurrent rollouts."""

    def __init__(self, pool_size: int, env_name: str | None = None, env_class: str | None = None, **env_kwargs: Any):
        self.pool_size = pool_size
        self.env_name = env_name
        self.env_class_path = env_class
        self.env_kwargs = env_kwargs
        self._env_class: type[RLEnvironment] | None = None
        self._actors: list[ray.actor.ActorHandle] = []
        self._available: asyncio.Queue[ray.actor.ActorHandle] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create all environment actors and call setup() on each."""
        if self._initialized:
            return

        self._env_class = get_env_class(env_name=self.env_name, env_class=self.env_class_path)
        actor_class = ray.remote(self._env_class)

        logger.info(f"Creating {self.pool_size} environment actors")
        self._actors = [actor_class.remote(**self.env_kwargs) for _ in range(self.pool_size)]

        logger.info("Running setup() on all environment actors...")
        setup_tasks = [actor.setup.remote() for actor in self._actors]
        try:
            await asyncio.to_thread(ray.get, setup_tasks)
        except Exception as e:
            logger.error(f"Error during environment setup, shutting down created actors: {e}")
            for actor in self._actors:
                ray.kill(actor)
            self._actors = []
            raise

        self._available = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)

        self._initialized = True
        logger.info(f"Environment pool initialized with {self.pool_size} actors")

    async def acquire(self) -> ray.actor.ActorHandle:
        """Acquire an available environment actor (blocks until available)."""
        if not self._initialized:
            raise RuntimeError("Pool not initialized")
        return await self._available.get()

    def release(self, actor: ray.actor.ActorHandle) -> None:
        """Release an environment actor back to the pool."""
        if self._available is not None:
            self._available.put_nowait(actor)

    async def shutdown(self) -> None:
        """Call shutdown() on all actors and terminate them."""
        if not self._initialized:
            return

        logger.info("Shutting down environment pool...")
        shutdown_tasks = [actor.shutdown.remote() for actor in self._actors]
        try:
            await asyncio.to_thread(ray.get, shutdown_tasks)
        except Exception as e:
            logger.warning(f"Error during environment shutdown: {e}")

        for actor in self._actors:
            ray.kill(actor)

        self._actors = []
        self._available = None
        self._initialized = False
        logger.info("Environment pool shutdown complete")

    @property
    def env_class(self) -> type[RLEnvironment] | None:
        return self._env_class

    def __len__(self) -> int:
        return self.pool_size
