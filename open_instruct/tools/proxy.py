"""
Tool proxy and actor for Ray-based tool execution.

This module provides a way to wrap tools in Ray actors, making them safe to pass
across processes. The flow is:

1. Create ToolActor from config: actor = create_tool_actor_from_config(...)
2. Wrap with ToolProxy: proxy = ToolProxy.from_actor(actor)
3. Pass proxy to vLLM engines

Usage:
    # Create actor from config (config.build() is called inside the actor)
    actor = create_tool_actor_from_config(
        config=serper_config,  # SerperSearchToolConfig instance
    )

    # Wrap with proxy
    proxy = ToolProxy.from_actor(actor)

    # Use proxy like a normal tool (async)
    result = await proxy(text="search query")
"""

from __future__ import annotations

import logging
from typing import Any

import ray

from open_instruct.tools.utils import Tool, ToolOutput

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

    This is the first step in the tool creation flow:
    1. Create actor: actor = create_tool_actor_from_config(config)
    2. Wrap with proxy: proxy = ToolProxy.from_actor(actor)

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


class ToolProxy(Tool):
    """Lightweight proxy that forwards calls to a ToolActor.

    This proxy is safe to send to other Ray actors because it only holds:
    - An actor handle (serializable)
    - Cached tool metadata (strings/dicts)

    All actual tool execution happens in the remote ToolActor.
    """

    def __init__(
        self,
        actor_handle: ray.actor.ActorHandle,
        tool_function_name: str,
        tool_description: str = "",
        tool_parameters: dict[str, Any] | None = None,
    ):
        """Initialize the proxy with an actor handle and cached metadata.

        Prefer using ToolProxy.from_actor() instead of calling this directly.

        Args:
            actor_handle: Handle to the ToolActor that owns the actual tool.
            tool_function_name: The function name for this tool (used for parsing).
            tool_description: Description of what the tool does.
            tool_parameters: JSON Schema for tool parameters.
        """
        self._actor = actor_handle
        self._tool_function_name = tool_function_name
        self._tool_description = tool_description
        self._tool_parameters = tool_parameters or {"type": "object", "properties": {}, "required": []}

    @property
    def tool_function_name(self) -> str:
        """Return the tool function name."""
        return self._tool_function_name

    @property
    def tool_description(self) -> str:
        """Return the tool description."""
        return self._tool_description

    @property
    def tool_parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        return self._tool_parameters

    async def __call__(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool via the remote actor (async)."""
        return await self._actor.call.remote(**kwargs)

    def get_stop_strings(self) -> list[str]:
        """Get stop strings from the remote tool."""
        return ray.get(self._actor.get_stop_strings.remote())

    def get_tool_names(self) -> list[str]:
        """Get tool names this tool exposes."""
        return [self._tool_function_name]

    def get_openai_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI format."""
        return ray.get(self._actor.get_openai_tool_definitions.remote())

    @classmethod
    def from_actor(cls, actor_handle: ray.actor.ActorHandle) -> ToolProxy:
        """Create a ToolProxy from an existing ToolActor.

        This is the second step in the tool creation flow:
        1. Create actor: actor = create_tool_actor_from_config(config)
        2. Wrap with proxy: proxy = ToolProxy.from_actor(actor)

        Args:
            actor_handle: Handle to a ToolActor.

        Returns:
            A ToolProxy that forwards calls to the actor.
        """
        # Fetch all metadata from actor in parallel
        name_ref = actor_handle.get_tool_function_name.remote()
        desc_ref = actor_handle.get_tool_description.remote()
        params_ref = actor_handle.get_tool_parameters.remote()

        tool_function_name, tool_description, tool_parameters = ray.get([name_ref, desc_ref, params_ref])

        return cls(
            actor_handle=actor_handle,
            tool_function_name=tool_function_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters,
        )
