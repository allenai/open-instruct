"""
Tool proxy and actor for Ray-based tool execution.

This module provides a way to wrap tools in Ray actors, making them safe to pass
across processes. The flow is:

1. Create ToolActor from config: actor = create_tool_actor_from_config(...)
2. Wrap with ToolProxy: proxy = ToolProxy.from_actor(actor)
3. Pass proxy to vLLM engines

Usage:
    # Create actor from config
    actor = create_tool_actor_from_config(
        class_path="open_instruct.tools.tools:SerperSearchTool",
        config=serper_config,
    )

    # Wrap with proxy
    proxy = ToolProxy.from_actor(actor)

    # Use proxy like a normal tool
    result = proxy(text="search query")
"""

from __future__ import annotations

import logging
from typing import Any

import ray

from open_instruct.tools.utils import Tool, ToolOutput

logger = logging.getLogger(__name__)

# Default max concurrency for tool actors
DEFAULT_MAX_CONCURRENCY = 512


def _import_from_path(class_path: str) -> type:
    """Import a class from a module:ClassName path."""
    import importlib

    module_name, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@ray.remote
class ToolActor:
    """Ray actor that holds and executes a tool.

    This actor constructs the tool inside the actor from a class path and config,
    avoiding the need to pickle tool objects with heavy dependencies (e.g., dr_agent).
    """

    def __init__(self, *, class_path: str, config: Any):
        """Initialize the actor by constructing a tool from config.

        Args:
            class_path: "module.submodule:ClassName" for lazy import.
            config: Config dataclass to pass to from_config().
        """
        tool_cls = _import_from_path(class_path)
        if not hasattr(tool_cls, "from_config"):
            raise ValueError(f"Tool class {tool_cls} does not have from_config method")

        self._tool = tool_cls.from_config(config)

        self.tool_function_name = self._tool.tool_function_name
        logger.info(f"ToolActor initialized for tool: {self.tool_function_name}")

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments."""
        return self._tool(**kwargs)

    def get_tool_function_name(self) -> str:
        """Get the tool's function name."""
        return self.tool_function_name

    def get_stop_strings(self) -> list[str]:
        """Get the tool's stop strings."""
        if hasattr(self._tool, "get_stop_strings") and callable(self._tool.get_stop_strings):
            return list(self._tool.get_stop_strings())
        return []

    def get_openai_tool_definition(self) -> dict:
        """Get the tool definition in OpenAI format for function calling."""
        return self._tool.get_openai_tool_definition()


def create_tool_actor_from_config(
    class_path: str, config: Any, max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any
) -> ray.actor.ActorHandle:
    """Create a ToolActor from a config.

    This is the first step in the tool creation flow:
    1. Create actor: actor = create_tool_actor_from_config(...)
    2. Wrap with proxy: proxy = ToolProxy.from_actor(actor)

    Args:
        class_path: "module.submodule:ClassName" for the tool class.
        config: Config dataclass to pass to from_config().
        max_concurrency: Maximum number of concurrent calls the actor can handle.
        **actor_options: Additional options to pass to ray.remote() for the actor.

    Returns:
        A Ray actor handle for the ToolActor.
    """
    options = {"max_concurrency": max_concurrency, **actor_options}
    actor_cls = ToolActor.options(**options)
    return actor_cls.remote(class_path=class_path, config=config)


class ToolProxy(Tool):
    """Lightweight proxy that forwards calls to a ToolActor.

    This proxy is safe to send to other Ray actors because it only holds:
    - An actor handle (serializable)
    - The tool function name (string)
    - The tool args (dict)

    All actual tool execution happens in the remote ToolActor.
    """

    def __init__(self, actor_handle: ray.actor.ActorHandle, tool_function_name: str):
        """Initialize the proxy with an actor handle.

        Prefer using ToolProxy.from_actor() instead of calling this directly.

        Args:
            actor_handle: Handle to the ToolActor that owns the actual tool.
            tool_function_name: The function name for this tool (used for parsing).
        """
        self._actor = actor_handle
        self._tool_function_name = tool_function_name

    @property
    def tool_function_name(self) -> str:
        """Return the tool function name."""
        return self._tool_function_name

    def __call__(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool via the remote actor."""
        return ray.get(self._actor.call.remote(**kwargs))

    def get_stop_strings(self) -> list[str]:
        """Get stop strings from the remote tool."""
        return ray.get(self._actor.get_stop_strings.remote())

    def get_openai_tool_definition(self) -> dict:
        """Get the tool definition in OpenAI format for function calling.

        Returns:
            Dict in OpenAI tool format with type, function name, description, and parameters.
        """
        return ray.get(self._actor.get_openai_tool_definition.remote())

    @classmethod
    def from_actor(cls, actor_handle: ray.actor.ActorHandle) -> ToolProxy:
        """Create a ToolProxy from an existing ToolActor.

        This is the second step in the tool creation flow:
        1. Create actor: actor = create_tool_actor_from_config(...)
        2. Wrap with proxy: proxy = ToolProxy.from_actor(actor)

        Args:
            actor_handle: Handle to a ToolActor.

        Returns:
            A ToolProxy that forwards calls to the actor.
        """
        # Fetch metadata from the actor
        tool_function_name = ray.get(actor_handle.get_tool_function_name.remote())
        return cls(actor_handle=actor_handle, tool_function_name=tool_function_name)
