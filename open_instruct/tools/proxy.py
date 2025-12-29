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
        class_path="open_instruct.tools.tools:DrAgentMCPTool",
        config=mcp_config,
        tool_name_override="snippet_search",
    )

    # Wrap with proxy
    proxy = ToolProxy.from_actor(actor)

    # Use proxy like a normal tool
    result = proxy(text="search query")
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

import ray

from open_instruct.tools.base import Tool, ToolOutput

if TYPE_CHECKING:
    pass

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

    def __init__(self, *, class_path: str, config: Any, tool_name_override: str | None = None):
        """Initialize the actor by constructing a tool from config.

        Args:
            class_path: "module.submodule:ClassName" for lazy import.
            config: Config dataclass to pass to from_config().
            tool_name_override: Override tool name for from_config() (used by MCP sub-tools).
        """
        tool_cls = _import_from_path(class_path)
        if not hasattr(tool_cls, "from_config"):
            raise ValueError(f"Tool class {tool_cls} does not have from_config method")

        # Check if from_config accepts tool_name_override
        sig = inspect.signature(tool_cls.from_config)
        if "tool_name_override" in sig.parameters and tool_name_override is not None:
            self._tool = tool_cls.from_config(config, tool_name_override=tool_name_override)
        else:
            self._tool = tool_cls.from_config(config)

        self.tool_function_name = self._tool.tool_function_name
        logger.info(f"ToolActor initialized for tool: {self.tool_function_name}")

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with the given arguments."""
        return self._tool(**kwargs)

    def get_tool_function_name(self) -> str:
        """Get the tool's function name."""
        return self.tool_function_name

    def get_tool_args(self) -> dict[str, Any]:
        """Get the tool's argument specification."""
        return self._tool.tool_args

    def get_stop_strings(self) -> list[str]:
        """Get the tool's stop strings."""
        if hasattr(self._tool, "get_stop_strings") and callable(self._tool.get_stop_strings):
            return list(self._tool.get_stop_strings())
        return []

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if text contains calls to the specified tool.

        This is used by DRTuluToolParser to detect tool calls.
        """
        if hasattr(self._tool, "mcp_tools"):
            for mcp_tool in self._tool.mcp_tools:
                if hasattr(mcp_tool, "tool_parser") and mcp_tool.tool_parser.has_calls(text, tool_name):
                    return True
        return False

    def get_mcp_tool_names(self) -> list[str]:
        """Get the names of MCP tools wrapped by this tool."""
        if hasattr(self._tool, "mcp_tools"):
            return [mcp_tool.name for mcp_tool in self._tool.mcp_tools]
        return []


def create_tool_actor_from_config(
    class_path: str,
    config: Any,
    tool_name_override: str | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    **actor_options: Any,
) -> ray.actor.ActorHandle:
    """Create a ToolActor from a config.

    This is the first step in the tool creation flow:
    1. Create actor: actor = create_tool_actor_from_config(...)
    2. Wrap with proxy: proxy = ToolProxy.from_actor(actor)

    Args:
        class_path: "module.submodule:ClassName" for the tool class.
        config: Config dataclass to pass to from_config().
        tool_name_override: Override tool name for from_config() (used by MCP sub-tools).
        max_concurrency: Maximum number of concurrent calls the actor can handle.
        **actor_options: Additional options to pass to ray.remote() for the actor.

    Returns:
        A Ray actor handle for the ToolActor.
    """
    options = {"max_concurrency": max_concurrency, **actor_options}
    actor_cls = ToolActor.options(**options)
    return actor_cls.remote(class_path=class_path, config=config, tool_name_override=tool_name_override)


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

        Prefer using ToolProxy.from_actor() instead of calling this directly.

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
        """Execute the tool via the remote actor."""
        return ray.get(self._actor.call.remote(**kwargs))

    def get_stop_strings(self) -> list[str]:
        """Get stop strings from the remote tool."""
        return ray.get(self._actor.get_stop_strings.remote())

    def has_calls(self, text: str, tool_name: str) -> bool:
        """Check if text contains calls to the specified tool."""
        return ray.get(self._actor.has_calls.remote(text, tool_name))

    def get_mcp_tool_names(self) -> list[str]:
        """Get the names of MCP tools wrapped by this tool."""
        return ray.get(self._actor.get_mcp_tool_names.remote())

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
        tool_args = ray.get(actor_handle.get_tool_args.remote())
        return cls(actor_handle=actor_handle, tool_function_name=tool_function_name, tool_args=tool_args)


# Keep these for backwards compatibility
def create_tool_proxy(tool: Tool, max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any) -> ToolProxy:
    """Create a ToolProxy from a tool instance.

    If the tool is already a ToolProxy, returns it unchanged.
    This is kept for backwards compatibility.

    Args:
        tool: The tool to wrap.
        max_concurrency: Maximum number of concurrent calls the actor can handle.
        **actor_options: Additional options for the Ray actor.

    Returns:
        A ToolProxy instance.
    """
    if isinstance(tool, ToolProxy):
        return tool
    # For backwards compatibility, create a simple wrapper actor
    # This path should rarely be used now that build_tools_from_config creates proxies directly
    raise NotImplementedError(
        "create_tool_proxy from tool instances is deprecated. "
        "Use create_tool_proxy_from_config or build_tools_from_config instead."
    )


def create_tool_proxies(
    tools: dict[str, Tool], max_concurrency: int = DEFAULT_MAX_CONCURRENCY, **actor_options: Any
) -> dict[str, ToolProxy]:
    """Create ToolProxy instances for all tools in a dictionary.

    If tools are already ToolProxy instances, returns them unchanged.

    Args:
        tools: Dictionary mapping tool names to Tool instances.
        max_concurrency: Maximum number of concurrent calls each actor can handle.
        **actor_options: Additional options to pass to all Tool actors.

    Returns:
        Dictionary mapping tool names to ToolProxy instances.
    """
    result = {}
    for name, tool in tools.items():
        if isinstance(tool, ToolProxy):
            result[name] = tool
        else:
            raise NotImplementedError(
                f"Tool '{name}' is not a ToolProxy. Use build_tools_from_config to create tools as proxies."
            )
    return result
