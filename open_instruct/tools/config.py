"""
Tool configuration system for composable tool arguments.

Each tool defines its own Config dataclass (in tools.py) with a from_config() classmethod.
This module composes them into ToolConfig for the main Args.
"""

from dataclasses import dataclass, field, fields
from typing import Literal

from open_instruct.tools.base import DRTuluToolParser, OpenInstructLegacyToolParser, Tool, ToolParser, VllmToolParser
from open_instruct.tools.tools import (
    MCP_TOOL_REGISTRY,
    MCPTool,
    MCPToolConfig,
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
    SearchTool,
    SearchToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
    YouSearchTool,
    YouSearchToolConfig,
)

# =============================================================================
# Tool Registry
# =============================================================================

# Maps tool names to (config_attr_name, tool_class, is_mcp_subtool)
# config_attr_name is the attribute on ToolConfig that holds this tool's config
TOOL_REGISTRY: dict[str, tuple[str, type[Tool], bool]] = {
    "code": ("python", PythonCodeTool, False),
    "python": ("python", PythonCodeTool, False),
    # Search tools - "search" defaults to Serper (Google Search)
    "search": ("serper_search", SerperSearchTool, False),
    "serper_search": ("serper_search", SerperSearchTool, False),
    "massive_ds_search": ("massive_ds_search", SearchTool, False),
    "s2_search": ("s2_search", S2SearchTool, False),
    "you_search": ("you_search", YouSearchTool, False),
    "mcp": ("mcp", MCPTool, False),
}

# Add MCP sub-tools - they use mcp config but with tool_name_override
for mcp_name in MCP_TOOL_REGISTRY:
    TOOL_REGISTRY[mcp_name] = ("mcp", MCPTool, True)

# Available parser types
PARSER_TYPES = Literal["legacy", "vllm", "dr_tulu"]


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "vllm", "dr_tulu"]


def _validate_registry_against_config(config_cls: type) -> None:
    """
    Validate that all config_attr entries in TOOL_REGISTRY have corresponding fields in ToolConfig.
    Called at module load time to catch misconfigurations early.
    """
    config_fields = {f.name for f in fields(config_cls)}
    required_attrs = {config_attr for config_attr, _, _ in TOOL_REGISTRY.values()}

    missing = required_attrs - config_fields
    if missing:
        raise ValueError(
            f"TOOL_REGISTRY references config attributes that don't exist in ToolConfig: {missing}. "
            f"Add these fields to ToolConfig or fix the registry."
        )


# =============================================================================
# Composed Tool Configuration
# =============================================================================


@dataclass
class ToolConfig:
    """
    Composed tool configuration that embeds individual tool configs.
    This gets nested in the main Args dataclass.

    CLI usage: --tool_config.tools search,code --tool_config.python.api_endpoint http://...
    """

    # General tool settings
    tools: list[str] | None = None
    """List of tools to enable. Available: code, search (Serper/Google), serper_search, massive_ds_search, s2_search, you_search, mcp, snippet_search, google_search, massive_serve, browse_webpage"""
    max_tool_calls: tuple[int, ...] = (5,)
    """Maximum number of tool calls allowed per generation."""
    mask_tool_use: bool = True
    """Whether to mask the tool output in training."""
    parser: str = "legacy"
    """Tool parser: 'legacy' (<tag>...</tag>), 'vllm' (native vLLM), or 'dr_tulu' (MCP-based)."""

    # Individual tool configurations (nested)
    # These must match the config_attr entries in TOOL_REGISTRY
    python: PythonCodeToolConfig = field(default_factory=PythonCodeToolConfig)
    """Python code execution tool configuration."""
    serper_search: SerperSearchToolConfig = field(default_factory=SerperSearchToolConfig)
    """Serper (Google Search) tool configuration. This is the default search tool."""
    massive_ds_search: SearchToolConfig = field(default_factory=SearchToolConfig)
    """Massive DS search tool configuration."""
    s2_search: S2SearchToolConfig = field(default_factory=S2SearchToolConfig)
    """Semantic Scholar search tool configuration."""
    you_search: YouSearchToolConfig = field(default_factory=YouSearchToolConfig)
    """You.com search tool configuration."""
    mcp: MCPToolConfig = field(default_factory=MCPToolConfig)
    """MCP tools configuration."""

    def __post_init__(self) -> None:
        """Validate that requested tools and parser exist."""
        if self.tools:
            available = get_available_tools()
            for tool in self.tools:
                if tool.lower() not in available:
                    raise ValueError(f"Unknown tool: {tool}. Available tools: {available}")

        available_parsers = get_available_parsers()
        if self.parser not in available_parsers:
            raise ValueError(f"Unknown parser: {self.parser}. Available parsers: {available_parsers}")


# Validate registry at module load time
_validate_registry_against_config(ToolConfig)


# =============================================================================
# Tool Setup Functions
# =============================================================================


@dataclass
class ToolSetup:
    """Result of setting up tools."""

    tools: dict[str, Tool]
    """Map of tool function name to tool instance."""
    parser: ToolParser | None
    """The tool parser to use."""
    stop_strings: list[str]
    """Stop strings for generation."""


def build_tools_from_config(config: ToolConfig, vllm_tool_parser=None, vllm_output_formatter=None) -> ToolSetup:
    """
    Build tools and parser from ToolConfig.

    Each tool is built using its Tool.from_config() classmethod.

    Args:
        config: The tool configuration with nested individual tool configs.
        vllm_tool_parser: Optional vLLM native tool parser (required if parser="vllm").
        vllm_output_formatter: Optional output formatter function (required if parser="vllm").

    Returns:
        ToolSetup with tools, parser, and stop strings.
    """
    if not config.tools:
        return ToolSetup(tools={}, parser=None, stop_strings=[])

    tools: dict[str, Tool] = {}
    tool_list: list[Tool] = []
    mcp_tools: list[Tool] = []

    for tool_name in config.tools:
        tool_name_lower = tool_name.lower()
        config_attr, tool_cls, is_mcp_subtool = TOOL_REGISTRY[tool_name_lower]
        tool_config = getattr(config, config_attr)

        # MCP sub-tools need the tool_name_override
        if is_mcp_subtool:
            tool = tool_cls.from_config(tool_config, tool_name_override=tool_name_lower)
        else:
            tool = tool_cls.from_config(tool_config)

        tools[tool.tool_function_name] = tool
        tool_list.append(tool)

        # Track MCP tools separately for DR Tulu parser
        if isinstance(tool, MCPTool):
            mcp_tools.append(tool)

    # Build parser based on type
    stop_strings: list[str] = []

    if config.parser == "legacy":
        parser = OpenInstructLegacyToolParser(tool_list=tool_list)
        stop_strings = parser.stop_sequences()

    elif config.parser == "vllm":
        # TODO: add support for native vllm tool outputs.
        # will require some extra work too.
        # how to handle multiple tool calls + generation prompt?
        if vllm_tool_parser is None or vllm_output_formatter is None:
            raise ValueError(
                "parser='vllm' requires vllm_tool_parser and vllm_output_formatter arguments. "
                "These should come from vLLM's native tool parsing setup."
            )
        parser = VllmToolParser(tool_parser=vllm_tool_parser, output_formatter=vllm_output_formatter)
        stop_strings = parser.stop_sequences()

    elif config.parser == "dr_tulu":
        if not mcp_tools:
            raise ValueError("parser='dr_tulu' requires at least one MCP tool to be configured.")
        parser = DRTuluToolParser(mcp_tool_list=mcp_tools)
        stop_strings = parser.stop_sequences()

    else:
        parser = None

    # For MCP tools, also get their stop strings (in addition to parser stop strings)
    for tool in tool_list:
        if isinstance(tool, MCPTool):
            stop_strings.extend(tool.get_stop_strings())

    return ToolSetup(tools=tools, parser=parser, stop_strings=list(set(stop_strings)))
