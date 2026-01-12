"""Tools module for open-instruct.

Main API:
- ToolArgs, ToolConfig: Configuration dataclasses
- build_tools_from_config: Build tools from config (returns Ray actor handles)
- create_tool_parser: Create a tool parser

For direct imports of specific tools/parsers, import from submodules:
- open_instruct.tools.tools (PythonCodeTool, SerperSearchTool, etc.)
- open_instruct.tools.parsers (OpenInstructLegacyToolParser, etc.)
- open_instruct.tools.proxy (ToolActor, create_tool_actor_from_config)
- open_instruct.tools.utils (Tool base class, ToolOutput, etc.)
"""

from open_instruct.tools.config import ToolArgs, ToolConfig, build_tools_from_config, create_tool_parser
from open_instruct.tools.parsers import ToolParser
from open_instruct.tools.proxy import ToolActor, create_tool_actor_from_config
from open_instruct.tools.utils import Tool, ToolOutput

__all__ = [
    # Config
    "ToolArgs",
    "ToolConfig",
    "build_tools_from_config",
    "create_tool_parser",
    # Types
    "Tool",
    "ToolOutput",
    "ToolParser",
    # Actor
    "ToolActor",
    "create_tool_actor_from_config",
]
