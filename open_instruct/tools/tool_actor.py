from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Optional

import ray

from open_instruct.tools.utils.tool_classes import Tool, ToolOutput

# Central registry mapping simple tool names to their import paths.
# We keep values as import paths to avoid importing heavy modules on the driver.
TOOL_CLASS_REGISTRY: Dict[str, str] = {
    # Lightweight search tools
    "search_s2": "open_instruct.tools.search_tool.search_tool:S2SearchTool",
    "search_you": "open_instruct.tools.search_tool.search_tool:YouSearchTool",
    "search_massive_ds": "open_instruct.tools.search_tool.search_tool:MassiveDSSearchTool",
    "search_serper": "open_instruct.tools.search_tool.search_tool:SerperSearchTool",
    # browse tool, for Ai2 internal use for now.
    "browse_crawl4ai": "open_instruct.tools.browse_tool.browse_tool:BrowseTool",
    # Code execution proxy tool (client)
    "code": "open_instruct.tools.python_tool.tool:PythonCodeTool",
}


def register_tool(name: str, class_path: str) -> None:
    """Register a tool by name.

    name: Unique identifier used by callers
    class_path: "module.submodule:ClassName" for lazy import inside the actor
    """
    TOOL_CLASS_REGISTRY[name] = class_path


def _import_from_path(class_path: str):
    module_name, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# NOTE: the max concurrency upper bounds the max number of tool calls.
# we override this when instantiating the actor in the main scripts.
@ray.remote(max_concurrency=512)
class ToolActor:
    """Generic Ray actor wrapper for any Tool subclass.

    Constructs the tool inside the actor from a class path and init kwargs.
    Exposes metadata and a call method. This avoids pickling live tool objects
    and isolates heavy resources.
    """

    def __init__(
        self,
        *,
        tool_name: Optional[str] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        class_path: Optional[str] = None,
    ):
        init_kwargs = init_kwargs or {}
        # Resolve class path via registry if tool_name is provided
        if tool_name is not None:
            resolved = TOOL_CLASS_REGISTRY.get(tool_name)
            if resolved is None:
                raise ValueError(f"Unknown tool_name '{tool_name}'. Registered: {list(TOOL_CLASS_REGISTRY.keys())}")
            class_path = resolved
        if not class_path:
            raise ValueError("Either tool_name or class_path must be provided")
        tool_cls = _import_from_path(class_path)
        # Filter kwargs to those accepted by the tool's constructor
        try:
            sig = inspect.signature(tool_cls.__init__)
            valid_params = set(sig.parameters.keys())
            # Never pass 'self'
            valid_params.discard("self")
            filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_params}
        except (TypeError, ValueError):
            # If signature introspection fails, fall back to provided kwargs
            filtered_kwargs = init_kwargs
        tool: Tool = tool_cls(**filtered_kwargs)
        self._tool = tool

    def get_start_str(self) -> str:
        return getattr(self._tool, "start_str", "")

    def get_stop_strings(self) -> List[str]:
        if hasattr(self._tool, "get_stop_strings") and callable(self._tool.get_stop_strings):
            return list(self._tool.get_stop_strings())
        end_str = getattr(self._tool, "end_str", "")
        return [end_str] if end_str else []

    def call(self, prompt: str) -> ToolOutput:
        return self._tool(prompt)
