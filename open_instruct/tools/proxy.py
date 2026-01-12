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
    name = ray.get(actor.get_tool_function_name.remote())
"""

from __future__ import annotations

import logging
from typing import Any

import ray

from open_instruct.tools.utils import ToolOutput

logger = logging.getLogger(__name__)

# Default max concurrency for tool actors
DEFAULT_MAX_CONCURRENCY = 512


@ray.remote
class ToolActor:
    """Ray actor that holds and executes a tool.

    This actor constructs the tool inside the actor from a config's build() method,
    avoiding the need to pickle tool objects with heavy dependencies (e.g., dr_agent).

    All tool calls are async - the actor awaits the tool's __call__ method.
    """

    def __init__(self, *, config: Any):
        """Initialize the actor by constructing a tool from config.

        Args:
            config: ToolConfig dataclass with a build() method that returns a Tool.
        """
        if not hasattr(config, "build"):
            raise ValueError(f"Config {type(config)} does not have a build() method")

        self._tool = config.build()

        self.tool_function_name = self._tool.tool_function_name
        logger.info(f"ToolActor initialized for tool: {self.tool_function_name}")

    async def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments (async)."""
        return await self._tool(**kwargs)

    def get_tool_function_name(self) -> str:
        """Get the tool's function name."""
        return self.tool_function_name

    def get_stop_strings(self) -> list[str]:
        """Get the tool's stop strings."""
        if hasattr(self._tool, "get_stop_strings") and callable(self._tool.get_stop_strings):
            return list(self._tool.get_stop_strings())
        return []

    def get_tool_names(self) -> list[str]:
        """Get tool names exposed by this tool."""
        return self._tool.get_tool_names()

    def get_openai_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI format."""
        return self._tool.get_openai_tool_definitions()

    def get_tool_description(self) -> str:
        """Get the tool's description."""
        return self._tool.tool_description

    def get_tool_parameters(self) -> dict:
        """Get the tool's parameter schema."""
        return self._tool.tool_parameters


def create_tool_actor_from_config(
    config: Any, max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any
) -> ray.actor.ActorHandle:
    """Create a ToolActor from a config.

    Args:
        config: ToolConfig dataclass with a build() method that returns a Tool.
        max_concurrency: Maximum number of concurrent calls the actor can handle.
        **actor_options: Additional options to pass to ray.remote() for the actor.

    Returns:
        A Ray actor handle for the ToolActor.
    """
    options = {"max_concurrency": max_concurrency, **actor_options}
    actor_cls = ToolActor.options(**options)
    return actor_cls.remote(config=config)
