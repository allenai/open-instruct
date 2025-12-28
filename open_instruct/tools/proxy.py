"""
Tool proxy and actor for Ray-based tool execution.

This module provides a way to wrap tools in Ray actors, making them safe to pass
across processes. The ToolProxy holds only an actor handle, which is serializable,
while the actual tool lives in the ToolActor.

Usage:
    # Create a tool
    tool = PythonCodeTool.from_config(config)

    # Wrap it in a Ray actor via proxy
    proxy = create_tool_proxy(tool)

    # Use proxy like a normal tool - calls are forwarded to the actor
    result = proxy(text="print('hello')")

    # Or wrap all tools in a dict
    tools = {"python": tool, "search": search_tool}
    proxied_tools = create_tool_proxies(tools)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import ray

from open_instruct.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default max concurrency for tool actors
DEFAULT_MAX_CONCURRENCY = 512


@ray.remote
class ToolActor:
    """Ray actor that holds and executes a tool.

    This actor owns the actual tool instance and executes calls on behalf of
    ToolProxy instances. By running the tool in an actor, we avoid serialization
    issues with non-picklable tool components (e.g., asyncio event loops, network
    connections, etc.).
    """

    def __init__(self, tool: Tool):
        """Initialize the actor with a tool instance.

        Args:
            tool: The tool instance to wrap. Must implement the Tool interface.
        """
        self.tool = tool
        self.tool_function_name = tool.tool_function_name
        logger.info(f"ToolActor initialized for tool: {self.tool_function_name}")

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments to pass to the tool's __call__ method.

        Returns:
            The ToolOutput from the tool execution.
        """
        return self.tool(**kwargs)

    def get_tool_function_name(self) -> str:
        """Get the tool's function name.

        Returns:
            The tool_function_name property of the wrapped tool.
        """
        return self.tool_function_name

    def get_tool_args(self) -> dict[str, Any]:
        """Get the tool's argument specification.

        Returns:
            The tool_args property of the wrapped tool.
        """
        return self.tool.tool_args


class ToolProxy(Tool):
    """Lightweight proxy that forwards calls to a ToolActor.

    This proxy is safe to send to other Ray actors because it only holds:
    - An actor handle (serializable)
    - The tool function name (string)
    - The tool args (dict)

    All actual tool execution happens in the remote ToolActor.
    """

    def __init__(self, actor_handle: ray.actor.ActorHandle, tool_function_name: str, tool_args: dict[str, Any]):
        """Initialize the proxy with an actor handle.

        Args:
            actor_handle: Handle to the ToolActor that owns the actual tool.
            tool_function_name: The function name for this tool (used for parsing).
            tool_args: The argument specification for this tool.
        """
        self._actor = actor_handle
        self._tool_function_name = tool_function_name
        self.tool_args = tool_args

    @property
    def tool_function_name(self) -> str:
        """Return the tool function name."""
        return self._tool_function_name

    def __call__(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool via the remote actor.

        Args:
            **kwargs: Arguments to pass to the tool.

        Returns:
            The ToolOutput from the remote tool execution.
        """
        return ray.get(self._actor.call.remote(**kwargs))

    @classmethod
    def from_tool(cls, tool: Tool, max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any) -> ToolProxy:
        """Create a ToolProxy from an existing tool.

        This is the recommended way to create a ToolProxy. It:
        1. Creates a ToolActor to hold the tool
        2. Returns a ToolProxy that forwards calls to the actor

        Args:
            tool: The tool to wrap in an actor.
            max_concurrency: Maximum number of concurrent calls the actor can handle.
                Defaults to 512.
            **actor_options: Additional options to pass to ray.remote() for the actor.
                Common options: num_cpus, num_gpus, resources, etc.

        Returns:
            A ToolProxy that forwards calls to the actor.

        Example:
            # Basic usage (uses default max_concurrency=512)
            proxy = ToolProxy.from_tool(my_tool)

            # With custom concurrency
            proxy = ToolProxy.from_tool(my_tool, max_concurrency=1000)

            # With other actor options
            proxy = ToolProxy.from_tool(my_tool, num_cpus=2)
        """
        # Merge max_concurrency with other actor options
        options = {"max_concurrency": max_concurrency, **actor_options}
        actor_cls = ToolActor.options(**options)
        actor_handle = actor_cls.remote(tool)

        # Get tool metadata from the original tool (avoids remote call)
        return cls(actor_handle=actor_handle, tool_function_name=tool.tool_function_name, tool_args=tool.tool_args)


def create_tool_proxy(tool: Tool, max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any) -> ToolProxy:
    """Create a ToolProxy from a tool.

    Convenience function that wraps ToolProxy.from_tool().

    Args:
        tool: The tool to wrap.
        max_concurrency: Maximum number of concurrent calls the actor can handle.
            Defaults to 512.
        **actor_options: Additional options for the Ray actor.

    Returns:
        A ToolProxy instance.
    """
    return ToolProxy.from_tool(tool, max_concurrency=max_concurrency, **actor_options)


def create_tool_proxies(
    tools: dict[str, Tool], max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any
) -> dict[str, ToolProxy]:
    """Create ToolProxy instances for all tools in a dictionary.

    Args:
        tools: Dictionary mapping tool names to Tool instances.
        max_concurrency: Maximum number of concurrent calls each actor can handle.
            Defaults to 512.
        **actor_options: Additional options to pass to all Tool actors.

    Returns:
        Dictionary mapping tool names to ToolProxy instances.

    Example:
        tools = {"python": python_tool, "search": search_tool}
        proxied_tools = create_tool_proxies(tools)

        # With custom concurrency
        proxied_tools = create_tool_proxies(tools, max_concurrency=1000)
    """
    return {
        name: create_tool_proxy(tool, max_concurrency=max_concurrency, **actor_options) for name, tool in tools.items()
    }
