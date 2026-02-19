"""Pool of Ray environment actors with acquire/release semantics."""

import collections
import logging
from typing import Any

import ray

from .base import get_env_class

logger = logging.getLogger(__name__)


@ray.remote
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is a Ray actor so it can be shared across multiple LLMRayActor engines.
    Acquire/release are called via .remote() from the vLLM generation loop.
    """

    def __init__(self, pool_size: int, env_name: str, **env_kwargs: Any):
        env_class = get_env_class(env_name)
        actor_class = ray.remote(env_class)

        logger.info(f"Creating {pool_size} '{env_name}' environment actors")
        self._actors = [actor_class.remote(**env_kwargs) for _ in range(pool_size)]

        setup_tasks = [actor.setup.remote() for actor in self._actors]
        try:
            ray.get(setup_tasks)
        except Exception as e:
            logger.warning(f"Error during environment setup: {e}")

        self._available: collections.deque[ray.actor.ActorHandle] = collections.deque(self._actors)
        logger.info(f"Environment pool ready: {pool_size} '{env_name}' actors")

    def acquire(self) -> ray.actor.ActorHandle | None:
        """Return an available actor, or None if all are in use."""
        if self._available:
            return self._available.popleft()
        return None

    def release(self, actor: ray.actor.ActorHandle) -> None:
        """Return an actor to the pool."""
        self._available.append(actor)

    def size(self) -> int:
        return len(self._actors)

    def available_count(self) -> int:
        return len(self._available)
