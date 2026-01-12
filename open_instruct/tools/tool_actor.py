"""
Tool actor for Ray-based tool execution.

This module provides a way to wrap tools in Ray actors.
This:
1. Allows us to pass ray handles instead of serializing tool objects
2. Allows us to make a single tool object handle all calls, making it easier to control things like concurrency.

Usage:
    # Create actor from config (config.build() is called inside the actor)
    actor = create_tool_actor_from_config(config=serper_config)

    # Call the tool via the actor
    result = await actor.call.remote(text="search query")

    # Get metadata
    name = ray.get(actor.get_call_name.remote())
"""

from typing import Any

import ray

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput, get_openai_tool_definitions

logger = logger_utils.setup_logger(__name__)

# Default max concurrency for tool actors
DEFAULT_MAX_CONCURRENCY = 512


@ray.remote
class ToolActor:
    """
    Ray actor that holds a tool.
    Essentially a proxy for the tool class itself.
    Separate from Tool classes as Ray actors don't allow inheritance.
    A few getter methods are provided to access tool metadata.
    """

    def __init__(self, *, config: BaseToolConfig):
        """Initialize the actor by constructing a tool from config.

        Args:
            config: BaseToolConfig dataclass with a build() method that returns a Tool.
        """
        self._tool: Tool = config.build()
        self.call_name = self._tool.call_name

    async def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments."""
        return await self._tool(**kwargs)

    def get_call_name(self) -> str:
        """Get the tool's call name (used when function calling)."""
        return self.call_name

    def get_openai_tool_definitions(self) -> dict[str, Any]:
        """Get tool definitions in OpenAI format."""
        return get_openai_tool_definitions(self._tool)

    def get_description(self) -> str:
        """Get the tool's description."""
        return self._tool.description

    def get_parameters(self) -> dict:
        """Get the tool's parameter schema."""
        return self._tool.parameters


def create_tool_actor_from_config(
    config: BaseToolConfig, max_concurrency: int = DEFAULT_MAX_CONCURRENCY
) -> ray.actor.ActorHandle:
    """Create a ToolActor from a config.

    Args:
        config: BaseToolConfig dataclass with a build() method that returns a Tool.
        max_concurrency: Maximum number of concurrent calls the actor can handle.

    Returns:
        A Ray actor handle for the ToolActor.
    """
    options: dict[str, Any] = {"max_concurrency": max_concurrency}
    return ray.remote(ToolActor).options(**options).remote(config=config)
