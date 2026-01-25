"""
Thin wrapper exposing EnvironmentPool as a Tool for TOOL_REGISTRY integration.
"""

from typing import Any

from open_instruct.environments.adapter import EnvironmentPool
from open_instruct.environments.base import StepResult
from open_instruct.tools.utils import Tool, ToolOutput


class EnvironmentTool(Tool):
    """Wraps EnvironmentPool as a Tool for TOOL_REGISTRY."""

    def __init__(
        self, pool: EnvironmentPool, config_name: str, call_name: str, description: str, parameters: dict[str, Any]
    ):
        super().__init__(config_name=config_name, description=description, call_name=call_name, parameters=parameters)
        self.pool = pool

    async def initialize(self):
        """Initialize the underlying pool (one-time setup)."""
        await self.pool.initialize()

    async def reset(self, request_id: str, prompt: str, info: dict) -> StepResult:
        """Called at start of rollout to acquire env from pool."""
        return await self.pool.acquire(request_id, prompt, info)

    async def execute(self, request_id: str | None = None, **kwargs: Any) -> ToolOutput:
        """Called when model invokes this tool."""
        result = await self.pool.step(request_id, **kwargs)
        return ToolOutput(output=result.observation, called=True, error="", timeout=False, runtime=0.0)

    def is_done(self, request_id: str) -> bool:
        """Check if env episode is complete."""
        return self.pool.is_done(request_id)

    def get_state(self, request_id: str) -> dict:
        """Get accumulated env state (rewards, step count, done)."""
        return self.pool.get_state(request_id)

    async def cleanup(self, request_id: str):
        """Release env back to pool after rollout."""
        await self.pool.release(request_id)
