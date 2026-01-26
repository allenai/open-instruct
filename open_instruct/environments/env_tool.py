"""
Thin wrapper exposing EnvironmentPool as a Tool for TOOL_REGISTRY integration.
"""

import importlib
import time
from typing import Any

from open_instruct.environments.adapter import EnvironmentPool
from open_instruct.environments.base import RLEnvironment, StepResult
from open_instruct.tools.utils import Tool, ToolOutput


def _import_class(fully_qualified_name: str) -> type:
    """Import a class from its fully qualified name (e.g., 'module.submodule.ClassName')."""
    module_path, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class EnvironmentTool(Tool):
    """Wraps EnvironmentPool as a Tool for TOOL_REGISTRY.

    This tool creates and manages an EnvironmentPool internally from config kwargs.
    """

    config_name: str = "environment"
    is_environment_tool: bool = True  # Flag for detection in vllm_utils

    def __init__(
        self,
        env_class: str,
        call_name: str,
        pool_size: int = 1,
        setup_fn: str | None = None,
        description: str = "RL environment tool",
        parameters: dict[str, Any] | None = None,
        **env_kwargs: Any,
    ):
        """Initialize EnvironmentTool from config.

        Args:
            env_class: Fully qualified class name of the RLEnvironment.
            call_name: Name used to invoke this tool.
            pool_size: Number of environment instances to pool.
            setup_fn: Optional fully qualified name of async setup function.
            description: Tool description.
            parameters: JSON schema for tool parameters.
            **env_kwargs: Additional kwargs passed to env_class constructor.
        """
        super().__init__(
            config_name=self.config_name,
            description=description,
            call_name=call_name,
            parameters=parameters or {"type": "object", "properties": {}},
        )

        # Import and store env class
        self._env_class: type[RLEnvironment] = _import_class(env_class)
        self._env_kwargs = env_kwargs

        # Import setup_fn if provided
        self._setup_fn = _import_class(setup_fn) if setup_fn else None

        # Create env factory that passes stored kwargs
        def env_factory(**runtime_kwargs: Any) -> RLEnvironment:
            return self._env_class(**{**self._env_kwargs, **runtime_kwargs})

        # Create the pool
        self.pool = EnvironmentPool(env_factory, pool_size, self._setup_fn)

    async def initialize(self):
        """Initialize the underlying pool (one-time setup)."""
        await self.pool.initialize()

    async def reset(self, request_id: str, prompt: str, info: dict) -> StepResult:
        """Called at start of rollout to acquire env from pool."""
        return await self.pool.acquire(request_id, prompt, info)

    async def execute(self, request_id: str | None = None, **kwargs: Any) -> ToolOutput:
        """Called when model invokes this tool."""
        start_time = time.perf_counter()
        result = await self.pool.step(request_id, **kwargs)
        runtime = time.perf_counter() - start_time
        return ToolOutput(output=result.observation, called=True, error="", timeout=False, runtime=runtime)

    def is_done(self, request_id: str) -> bool:
        """Check if env episode is complete."""
        return self.pool.is_done(request_id)

    def get_state(self, request_id: str) -> dict:
        """Get accumulated env state (rewards, step count, done)."""
        return self.pool.get_state(request_id)

    async def cleanup(self, request_id: str):
        """Release env back to pool after rollout."""
        await self.pool.release(request_id)
