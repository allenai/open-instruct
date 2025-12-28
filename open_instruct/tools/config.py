"""
Tool configuration system for composable tool arguments.

Each tool defines its own Config dataclass (in tools.py) with a from_config() classmethod.
This module provides:
- ToolArgs: flat dataclass for CLI argument parsing (HfArgumentParser compatible)
- ToolConfig: internal structured config used by build_tools_from_config
"""

import logging
from dataclasses import dataclass, field, fields
from typing import Literal

from open_instruct.tools.base import DRTuluToolParser, OpenInstructLegacyToolParser, Tool, ToolParser, VllmToolParser
from open_instruct.tools.tools import (
    MCP_TOOL_REGISTRY,
    DrAgentMCPTool,
    DrAgentMCPToolConfig,
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

logger = logging.getLogger(__name__)

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
    "mcp": ("mcp", DrAgentMCPTool, False),
}

# Add MCP sub-tools - they use mcp config but with tool_name_override
for mcp_name in MCP_TOOL_REGISTRY:
    TOOL_REGISTRY[mcp_name] = ("mcp", DrAgentMCPTool, True)

# Available parser types
PARSER_TYPES = Literal["legacy", "vllm", "dr_tulu"]


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "vllm", "dr_tulu"]


# =============================================================================
# ToolArgs - Flat dataclass for HfArgumentParser
# =============================================================================


@dataclass
class ToolArgs:
    """
    Flat tool arguments for CLI parsing with HfArgumentParser.

    This dataclass is passed to ArgumentParserPlus alongside Args, TokenizerConfig, etc.
    Use to_tool_config() to convert to the internal ToolConfig structure.
    """

    # General tool settings
    tools: list[str] | None = None
    """List of tools to enable. Available: code, search, serper_search, massive_ds_search, s2_search, you_search, mcp."""
    max_tool_calls: int = 5
    """Maximum number of tool calls allowed per generation."""
    mask_tool_use: bool = True
    """Whether to mask the tool output in training."""
    tool_parser: str = "legacy"
    """Tool parser: 'legacy' (<tag>...</tag>), 'vllm' (native vLLM), or 'dr_tulu' (MCP-based)."""
    tool_tag_names: list[str] | None = None
    """Override the XML tag names for tools, in the same order as --tools.

    Allows using consistent tags while swapping backend implementations.
    For example: --tools s2_search code --tool_tag_names search python
    This uses Semantic Scholar with <search> tag and code tool with <python> tag.
    Must have the same length as --tools if provided.
    """

    # Code tool settings
    code_api_endpoint: str | None = None
    """API endpoint for code execution tool."""
    code_timeout_seconds: int = 3
    """Timeout for code execution in seconds."""

    # Massive DS search settings
    search_api_endpoint: str | None = None
    """API endpoint for massive_ds search tool."""
    search_num_documents: int = 3
    """Number of documents to return from massive_ds search."""

    # Serper (Google) search settings
    serper_num_results: int = 5
    """Number of results from Serper (Google) search."""

    # Semantic Scholar settings
    s2_num_results: int = 10
    """Number of results from Semantic Scholar."""

    # You.com settings
    you_num_results: int = 10
    """Number of results from You.com."""

    def __post_init__(self) -> None:
        """Validate tool arguments."""
        if self.tools:
            available = get_available_tools()
            for tool in self.tools:
                if tool.lower() not in available:
                    raise ValueError(f"Unknown tool: {tool}. Available tools: {available}")

        available_parsers = get_available_parsers()
        if self.tool_parser not in available_parsers:
            raise ValueError(f"Unknown parser: {self.tool_parser}. Available parsers: {available_parsers}")

        # Validate tool_tag_names length matches tools
        if self.tool_tag_names is not None:
            if not self.tools:
                raise ValueError("--tool_tag_names requires --tools to be specified")
            if len(self.tool_tag_names) != len(self.tools):
                raise ValueError(
                    f"--tool_tag_names must have the same length as --tools. "
                    f"Got {len(self.tool_tag_names)} tag names for {len(self.tools)} tools."
                )

    def to_tool_config(self) -> "ToolConfig":
        """Convert flat ToolArgs to internal ToolConfig structure."""
        return ToolConfig(
            tools=self.tools,
            max_tool_calls=self.max_tool_calls,
            mask_tool_use=self.mask_tool_use,
            parser=self.tool_parser,
            tool_tag_names=self.tool_tag_names,
            python=PythonCodeToolConfig(
                api_endpoint=self.code_api_endpoint, timeout_seconds=self.code_timeout_seconds
            ),
            serper_search=SerperSearchToolConfig(number_of_results=self.serper_num_results),
            massive_ds_search=SearchToolConfig(
                api_endpoint=self.search_api_endpoint, number_documents=self.search_num_documents
            ),
            s2_search=S2SearchToolConfig(number_of_results=self.s2_num_results),
            you_search=YouSearchToolConfig(number_of_results=self.you_num_results),
        )


# =============================================================================
# ToolConfig - Internal structured configuration
# =============================================================================


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


@dataclass
class ToolConfig:
    """
    Internal structured tool configuration.

    This is created from ToolArgs.to_tool_config() and used by build_tools_from_config().
    """

    # General tool settings
    tools: list[str] | None = None
    max_tool_calls: int = 5
    mask_tool_use: bool = True
    parser: str = "legacy"
    tool_tag_names: list[str] | None = None
    """Tag name overrides for each tool, in the same order as tools list."""

    # Individual tool configurations (nested)
    python: PythonCodeToolConfig = field(default_factory=PythonCodeToolConfig)
    serper_search: SerperSearchToolConfig = field(default_factory=SerperSearchToolConfig)
    massive_ds_search: SearchToolConfig = field(default_factory=SearchToolConfig)
    s2_search: S2SearchToolConfig = field(default_factory=S2SearchToolConfig)
    you_search: YouSearchToolConfig = field(default_factory=YouSearchToolConfig)
    mcp: DrAgentMCPToolConfig = field(default_factory=DrAgentMCPToolConfig)


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
    tool_mappings: list[str] = []  # For logging

    for i, tool_name in enumerate(config.tools):
        tool_name_lower = tool_name.lower()
        config_attr, tool_cls, is_mcp_subtool = TOOL_REGISTRY[tool_name_lower]
        tool_config = getattr(config, config_attr)

        # Apply tag_name from the parallel list if provided
        if config.tool_tag_names and hasattr(tool_config, "tag_name"):
            tool_config.tag_name = config.tool_tag_names[i]

        # MCP sub-tools need the tool_name_override
        if is_mcp_subtool:
            tool = tool_cls.from_config(tool_config, tool_name_override=tool_name_lower)
        else:
            tool = tool_cls.from_config(tool_config)

        tools[tool.tool_function_name] = tool
        tool_list.append(tool)

        # Log the mapping
        tag_name = tool.tool_function_name
        tool_mappings.append(f"{tool_name} -> <{tag_name}>")

        # Track MCP tools separately for DR Tulu parser
        if isinstance(tool, DrAgentMCPTool):
            mcp_tools.append(tool)

    # Log tool configuration
    logger.info(f"Configured {len(tools)} tool(s): {', '.join(tool_mappings)}")

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
        assert len(mcp_tools) == 1, "DR Tulu only uses the MCP tool."
        assert len(tool_list) == 1, "DR Tulu only uses the MCP tool. Remove other tools."
        parser = DRTuluToolParser(mcp_tool_list=mcp_tools)
        stop_strings = parser.stop_sequences()

    else:
        parser = None

    # For MCP tools, also get their stop strings (in addition to parser stop strings)
    for tool in tool_list:
        if isinstance(tool, DrAgentMCPTool):
            stop_strings.extend(tool.get_stop_strings())

    return ToolSetup(tools=tools, parser=parser, stop_strings=list(set(stop_strings)))
