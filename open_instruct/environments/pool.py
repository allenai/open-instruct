"""
Environment pool for managing Ray environment actors.

The pool maintains a set of environment actor handles and provides
acquire/release semantics for rollout workers.
"""

import asyncio
import logging
from typing import Any

import ray

from .base import RLEnvironment, get_env_class, make_env_actor

logger = logging.getLogger(__name__)


class EnvironmentPool:
    """
    Pool of RLEnvironment Ray actors.

    The pool creates a fixed number of environment actors and provides
    acquire/release semantics for concurrent rollouts. Each actor can
    handle one episode at a time.

    Usage:
        pool = EnvironmentPool(
            env_name="wordle",  # or env_class="mymodule.MyEnv"
            pool_size=64,
            backend="e2b",  # passed to env __init__
        )
        await pool.initialize()

        # In rollout loop
        env_actor = await pool.acquire()
        try:
            result = await env_actor.reset.remote(task_id="task_123")
            # ... run episode ...
        finally:
            pool.release(env_actor)

        # Cleanup
        await pool.shutdown()
    """

    def __init__(
        self,
        pool_size: int,
        env_name: str | None = None,
        env_class: str | None = None,
        **env_kwargs: Any,
    ):
        """
        Initialize the environment pool.

        Args:
            pool_size: Number of environment actors to create
            env_name: Name of registered environment
            env_class: Full class path for unregistered environment
            **env_kwargs: Arguments passed to environment __init__
        """
        self.pool_size = pool_size
        self.env_name = env_name
        self.env_class_path = env_class
        self.env_kwargs = env_kwargs

        self._env_class: type[RLEnvironment] | None = None
        self._actors: list[ray.actor.ActorHandle] = []
        self._available: asyncio.Queue[ray.actor.ActorHandle] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Create all environment actors.

        Must be called before using acquire/release.
        """
        if self._initialized:
            return

        # Get the environment class
        self._env_class = get_env_class(env_name=self.env_name, env_class=self.env_class_path)

        # Create Ray actor class from environment class
        actor_class = make_env_actor(self._env_class)

        # Create actors
        logger.info(
            f"Creating {self.pool_size} environment actors " f"(env_name={self.env_name}, env_class={self.env_class_path})"
        )

        self._actors = [actor_class.remote(**self.env_kwargs) for _ in range(self.pool_size)]

        # Initialize availability queue
        self._available = asyncio.Queue()
        for actor in self._actors:
            self._available.put_nowait(actor)

        self._initialized = True
        logger.info(f"Environment pool initialized with {self.pool_size} actors")

    async def acquire(self) -> ray.actor.ActorHandle:
        """
        Acquire an available environment actor.

        Blocks until an actor is available.

        Returns:
            Ray actor handle for an RLEnvironment
        """
        if not self._initialized:
            raise RuntimeError("Pool not initialized. Call initialize() first.")
        return await self._available.get()

    def release(self, actor: ray.actor.ActorHandle) -> None:
        """
        Release an environment actor back to the pool.

        Args:
            actor: The actor handle to release
        """
        if self._available is not None:
            self._available.put_nowait(actor)

    async def shutdown(self) -> None:
        """
        Shutdown all environment actors.

        Calls close() on each actor and terminates them.
        """
        if not self._initialized:
            return

        logger.info("Shutting down environment pool...")

        # Call close() on all actors
        close_tasks = [actor.close.remote() for actor in self._actors]
        try:
            await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in close_tasks], return_exceptions=True)
        except Exception as e:
            logger.warning(f"Error closing some actors: {e}")

        # Kill actors
        for actor in self._actors:
            ray.kill(actor)

        self._actors = []
        self._available = None
        self._initialized = False
        logger.info("Environment pool shutdown complete")

    @property
    def env_class(self) -> type[RLEnvironment] | None:
        """Get the environment class (available after initialize())."""
        return self._env_class

    def __len__(self) -> int:
        """Return the pool size."""
        return self.pool_size
