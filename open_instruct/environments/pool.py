"""Pool of Ray actors with acquire/release semantics."""

import collections
from typing import Any

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@ray.remote
class EnvironmentPool:
    """Shared pool of RLEnvironment Ray actors for concurrent rollouts.

    This is a Ray actor so it can be shared across multiple LLMRayActor engines.
    Acquire/release are called via .remote() from the vLLM generation loop.
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

        self._available: collections.deque[ray.actor.ActorHandle] = collections.deque(self._actors)
        logger.info(f"Pool ready: {pool_size} {actor_class.__name__} actors")

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
