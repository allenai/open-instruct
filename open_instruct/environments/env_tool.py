"""
Thin wrapper exposing EnvironmentPool as a Tool for TOOL_REGISTRY integration.
"""

import importlib
import time
from typing import Any

from open_instruct import logger_utils
from open_instruct.environments.adapter import EnvironmentPool
from open_instruct.environments.base import RLEnvironment, StepResult
from open_instruct.tools.utils import Tool, ToolOutput

logger = logger_utils.setup_logger(__name__)


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
        # Set default parameters based on environment type if not provided
        if parameters is None:
            env_name = env_kwargs.get("env_name", "")
            if "wordle" in env_name.lower():
                # Default Wordle parameters
                parameters = {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string", "description": "Your 5-letter guess for the Wordle game"}
                    },
                    "required": ["word"],
                }
            elif "appworld" in env_class.lower():
                # Default AppWorld parameters
                parameters = {
                    "type": "object",
                    "properties": {
                        "api_code": {
                            "type": "string",
                            "description": "Python code to execute in the AppWorld environment. Use the AppWorld API to interact with the task.",
                        }
                    },
                    "required": ["api_code"],
                }
            else:
                # Generic empty schema
                parameters = {"type": "object", "properties": {}}

        super().__init__(
            config_name=self.config_name, description=description, call_name=call_name, parameters=parameters
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

    async def reset(self, _request_id: str, info: dict) -> StepResult:
        """Called at start of rollout to acquire env from pool."""
        logger.info(f"[EnvironmentTool] reset called for {_request_id}")
        result = await self.pool.acquire(_request_id, info)
        logger.info(
            f"[EnvironmentTool] reset done for {_request_id}: {result.observation[:100] if result.observation else ''}"
        )
        return result

    async def execute(self, _request_id: str, **kwargs: Any) -> ToolOutput:
        """Called when model invokes this tool."""
        logger.info(f"[EnvironmentTool] execute called for {_request_id} with kwargs: {kwargs}")
        start_time = time.perf_counter()
        result = await self.pool.step(_request_id, **kwargs)
        runtime = time.perf_counter() - start_time
        logger.info(f"[EnvironmentTool] execute done for {_request_id}, runtime={runtime:.2f}s, done={result.done}")
        return ToolOutput(output=result.observation, called=True, error="", timeout=False, runtime=runtime)

    async def safe_execute(self, _request_id: str, **kwargs: Any) -> ToolOutput:
        """Environment tools need _request_id, so don't strip it."""
        return await self.execute(_request_id, **kwargs)

    def is_done(self, _request_id: str) -> bool:
        """Check if env episode is complete."""
        return self.pool.is_done(_request_id)

    def get_state(self, _request_id: str) -> dict:
        """Get accumulated env state (rewards, step count, done)."""
        return self.pool.get_state(_request_id)

    async def cleanup(self, _request_id: str):
        """Release env back to pool after rollout."""
        logger.info(f"[EnvironmentTool] cleanup called for {_request_id}")
        await self.pool.release(_request_id)
        logger.info(f"[EnvironmentTool] cleanup done for {_request_id}")

    def get_primary_param_name(self) -> str:
        """Get the primary parameter name from the tool schema.

        Returns the first required parameter name, or 'content' as fallback.
        """
        required = self.parameters.get("required", [])
        if required and len(required) > 0:
            return required[0]
        # Fallback to first property if no required params
        properties = self.parameters.get("properties", {})
        if properties:
            return next(iter(properties.keys()))
        return "content"  # Ultimate fallback
