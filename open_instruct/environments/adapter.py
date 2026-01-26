"""
Adapter layer bridging open-instruct data format to pure RLEnvironment instances.

EnvironmentAdapter: Bridges dataset info to env, tracks rewards across steps.
EnvironmentPool: Manages pool of adapters for concurrent rollouts.
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from open_instruct.environments.base import RLEnvironment, StepResult


class EnvironmentAdapter:
    """Bridges open-instruct data format to pure RLEnvironment."""

    def __init__(self, env_factory: Callable[..., RLEnvironment]):
        self.env_factory = env_factory
        self.env: RLEnvironment | None = None
        self.rewards: list[float] = []
        self.step_count = 0
        self.done = False

    async def setup(self, prompt: str, info: dict[str, Any]) -> StepResult:
        """Create env from dataset info, then reset it."""
        self.rewards = []
        self.step_count = 0
        self.done = False

        env_kwargs = info["env_config"]
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
        self, env_factory: Callable[..., RLEnvironment], pool_size: int, setup_fn: Callable[[], Any] | None = None
    ):
        self.env_factory = env_factory
        self.pool_size = pool_size
        self.setup_fn = setup_fn
        self._pool: asyncio.Queue[EnvironmentAdapter] = asyncio.Queue()
        self._active: dict[str, EnvironmentAdapter] = {}  # request_id -> adapter

    async def initialize(self):
        """One-time setup (e.g., spawn AppWorld servers)."""
        if self.setup_fn:
            await self.setup_fn()
        for _ in range(self.pool_size):
            await self._pool.put(EnvironmentAdapter(self.env_factory))

    async def acquire(self, request_id: str, prompt: str, info: dict) -> StepResult:
        """Get adapter from pool, setup for this request."""
        adapter = await self._pool.get()
        self._active[request_id] = adapter
        return await adapter.setup(prompt, info)

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
