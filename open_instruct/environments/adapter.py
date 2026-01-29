"""
Adapter layer bridging open-instruct data format to pure RLEnvironment instances.

EnvironmentAdapter: Bridges dataset info to env, tracks rewards across steps.
EnvironmentPool: Manages pool of adapters for concurrent rollouts.
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from open_instruct import logger_utils
from open_instruct.environments.base import RLEnvironment, StepResult

logger = logger_utils.setup_logger(__name__)


class EnvironmentAdapter:
    """Bridges open-instruct data format to pure RLEnvironment."""

    def __init__(self, env_factory: Callable[..., RLEnvironment], server_url: str | None = None):
        self.env_factory = env_factory
        self.server_url = server_url  # Server URL for this adapter (if applicable)
        self.env: RLEnvironment | None = None
        self.rewards: list[float] = []
        self.step_count = 0
        self.done = False

    async def setup(self, info: dict[str, Any]) -> StepResult:
        """Create env from dataset info, then reset it.

        Args:
            info: Dict containing 'env_config' with kwargs for the environment factory.
        """
        self.rewards = []
        self.step_count = 0
        self.done = False

        env_kwargs = info["env_config"].copy()
        # Inject server_url if available and not already in env_kwargs
        if self.server_url and "server_url" not in env_kwargs:
            env_kwargs["server_url"] = self.server_url
        self.env = self.env_factory(**env_kwargs)
        result = await self._maybe_await(self.env.reset())
        self.rewards.append(result.reward)
        self.done = result.done
        return result

    async def step(self, **action_kwargs) -> StepResult:
        """Execute action on env."""
        result = await self._maybe_await(self.env.step(action_kwargs))
        self.rewards.append(result.reward)
        self.step_count += 1
        self.done = result.done
        return result

    async def _maybe_await(self, result):
        """Handle both sync and async env methods."""
        if inspect.isawaitable(result):
            return await result
        return result

    def get_state(self) -> dict:
        return {"rewards": self.rewards, "step_count": self.step_count, "done": self.done}

    def cleanup(self):
        self.env.close()
        self.env = None


class EnvironmentPool:
    """Manages pool of EnvironmentAdapters for concurrent rollouts."""

    def __init__(
        self,
        env_factory: Callable[..., RLEnvironment],
        pool_size: int,
        setup_fn: Callable[[], Any] | Callable[[int], Any] | None = None,
    ):
        self.env_factory = env_factory
        self.pool_size = pool_size
        self.setup_fn = setup_fn
        self._pool: asyncio.Queue[EnvironmentAdapter] = asyncio.Queue()
        self._active: dict[str, EnvironmentAdapter] = {}  # request_id -> adapter
        self._server_urls: list[str] = []  # Server URLs from setup_fn

    async def initialize(self):
        """One-time setup (e.g., spawn AppWorld servers)."""
        if self.setup_fn:
            # Check if setup_fn accepts pool_size parameter for backward compatibility
            sig = inspect.signature(self.setup_fn)
            if len(sig.parameters) > 0:
                # New style: setup_fn accepts pool_size
                result = await self.setup_fn(self.pool_size)
            else:
                # Old style: setup_fn takes no parameters
                result = await self.setup_fn()
            # If setup_fn returns server URLs, store them
            if result and isinstance(result, list):
                self._server_urls = result
        for i in range(self.pool_size):
            # Pass server_url to adapter if available
            server_url = self._server_urls[i] if i < len(self._server_urls) else None
            await self._pool.put(EnvironmentAdapter(self.env_factory, server_url=server_url))

    async def acquire(self, request_id: str, info: dict) -> StepResult:
        """Get adapter from pool, setup for this request."""
        if request_id in self._active:
            raise ValueError(f"request_id '{request_id}' is already active. Each request must have a unique ID.")
        adapter = await self._pool.get()
        self._active[request_id] = adapter
        return await adapter.setup(info)

    async def step(self, request_id: str, **action) -> StepResult:
        return await self._active[request_id].step(**action)

    def get_state(self, request_id: str) -> dict:
        return self._active[request_id].get_state()

    def is_done(self, request_id: str) -> bool:
        return self._active[request_id].done

    async def release(self, request_id: str):
        """Return adapter to pool."""
        adapter = self._active.pop(request_id)
        adapter.cleanup()
        await self._pool.put(adapter)
