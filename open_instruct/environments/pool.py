"""Pool of Ray environment actors with acquire/release semantics."""

from queue import Queue
from typing import Any

import ray

from open_instruct import logger_utils

from .base import RLEnvironment, get_env_class

logger = logger_utils.setup_logger(__name__)


class EnvironmentPool:
    """Pool of RLEnvironment Ray actors for concurrent rollouts."""

    def __init__(self, pool_size: int, env_name: str, **env_kwargs: Any):
        self.pool_size = pool_size
        self.env_class = get_env_class(env_name)
        actor_class = ray.remote(self.env_class)

        logger.info(f"Creating {pool_size} {env_name} environment actors")
        self._actors = [actor_class.remote(**env_kwargs) for _ in range(pool_size)]

        logger.info("Running setup() on all environment actors...")
        try:
            ray.get([actor.setup.remote() for actor in self._actors])
        except Exception as e:
            logger.error(f"Error during environment setup, shutting down actors: {e}")
            for actor in self._actors:
                ray.kill(actor)
            self._actors = []
            raise

        self._available: Queue[ray.actor.ActorHandle] = Queue()
        for actor in self._actors:
            self._available.put(actor)

        logger.info(f"Environment pool ready with {pool_size} actors")

    def acquire(self) -> ray.actor.ActorHandle:
        """Acquire an available environment actor (blocks until available)."""
        return self._available.get()

    def release(self, actor: ray.actor.ActorHandle) -> None:
        """Release an environment actor back to the pool."""
        self._available.put(actor)

    def shutdown(self) -> None:
        """Call shutdown() on all actors and terminate them."""
        logger.info("Shutting down environment pool...")
        try:
            ray.get([actor.shutdown.remote() for actor in self._actors])
        except Exception as e:
            logger.warning(f"Error during environment shutdown: {e}")

        for actor in self._actors:
            ray.kill(actor)
        self._actors = []
        logger.info("Environment pool shutdown complete")

    def __len__(self) -> int:
        return self.pool_size
