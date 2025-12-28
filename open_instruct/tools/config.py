"""
Tool configuration system for composable tool arguments.

Each tool defines its own Config dataclass (in tools.py) with:
- A `cli_prefix` ClassVar that defines the CLI argument prefix
- Fields that become CLI arguments as `--{prefix}{field_name}`

This module provides:
- ToolArgs: flat dataclass for CLI argument parsing (HfArgumentParser compatible)
- ToolConfig: internal structured config used by build_tools_from_config

ToolArgs is built automatically from individual tool configs.
"""

import logging
from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from typing import Any, Literal

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
# Tool Config Registry
# =============================================================================

# Maps config attribute name (used in ToolConfig) to config class
# Each config class must have a `cli_prefix` ClassVar
TOOL_CONFIG_REGISTRY: dict[str, type] = {
    "python": PythonCodeToolConfig,
    "massive_ds_search": SearchToolConfig,
    "serper_search": SerperSearchToolConfig,
    "s2_search": S2SearchToolConfig,
    "you_search": YouSearchToolConfig,
    "mcp": DrAgentMCPToolConfig,
}

# Fields to exclude from CLI exposure (common across all configs)
EXCLUDED_FIELDS = {"tag_name", "cli_prefix"}

# =============================================================================
# Tool Registry
# =============================================================================

# Maps tool names to (config_attr_name, tool_class, is_mcp_subtool)
TOOL_REGISTRY: dict[str, tuple[str, type[Tool], bool]] = {
    "code": ("python", PythonCodeTool, False),
    "python": ("python", PythonCodeTool, False),
    "search": ("serper_search", SerperSearchTool, False),
    "serper_search": ("serper_search", SerperSearchTool, False),
    "massive_ds_search": ("massive_ds_search", SearchTool, False),
    "s2_search": ("s2_search", S2SearchTool, False),
    "you_search": ("you_search", YouSearchTool, False),
    "mcp": ("mcp", DrAgentMCPTool, False),
}

# Add MCP sub-tools
for mcp_name in MCP_TOOL_REGISTRY:
    TOOL_REGISTRY[mcp_name] = ("mcp", DrAgentMCPTool, True)

PARSER_TYPES = Literal["legacy", "vllm", "dr_tulu"]


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "vllm", "dr_tulu"]


# =============================================================================
# Dynamic ToolArgs Construction
# =============================================================================


def _get_field_default(f: Any) -> Any:
    """Extract the default value or factory from a dataclass field."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return field(default_factory=f.default_factory)
    return MISSING


def _build_tool_args() -> tuple[type, dict[str, tuple[str, str]]]:
    """
    Build ToolArgs dataclass from tool config classes.

    Returns:
        Tuple of (ToolArgs class, mapping from CLI field to (config_attr, config_field))
    """
    all_fields: list[tuple[str, type, Any]] = []
    field_mapping: dict[str, tuple[str, str]] = {}
    seen_cli_names: dict[str, str] = {}  # cli_name -> "config_attr.field_name" for error messages

    # Base fields
    base_fields = [
        ("tools", list[str] | None, field(default=None)),
        ("max_tool_calls", int, field(default=5)),
        ("mask_tool_use", bool, field(default=True)),
        ("tool_parser", str, field(default="legacy")),
        ("tool_tag_names", list[str] | None, field(default=None)),
    ]
    all_fields.extend(base_fields)

    # Add fields from each tool config
    for config_attr, config_cls in TOOL_CONFIG_REGISTRY.items():
        prefix = getattr(config_cls, "cli_prefix", "")

        for f in fields(config_cls):
            if f.name in EXCLUDED_FIELDS:
                continue

            cli_name = f"{prefix}{f.name}"

            # Check for clashes
            if cli_name in seen_cli_names:
                raise ValueError(
                    f"CLI argument clash: --{cli_name} is used by both "
                    f"{seen_cli_names[cli_name]} and {config_attr}.{f.name}. "
                    f"Change one of the field names or cli_prefix values to avoid this."
                )
            seen_cli_names[cli_name] = f"{config_attr}.{f.name}"

            # Get default
            default = _get_field_default(f)
            if default is MISSING:
                all_fields.append((cli_name, f.type, field()))
            elif isinstance(default, field().__class__):
                all_fields.append((cli_name, f.type, default))
            else:
                all_fields.append((cli_name, f.type, field(default=default)))

            field_mapping[cli_name] = (config_attr, f.name)

    # Create class
    cls = make_dataclass(
        "ToolArgs",
        all_fields,
        namespace={
            "__doc__": """
Flat tool arguments for CLI parsing with HfArgumentParser.

Auto-generated from tool config classes. Each tool config's fields are
exposed with its cli_prefix prepended.

Use to_tool_config() to convert to the internal ToolConfig structure.
""",
            "_field_mapping": field_mapping,
            "__post_init__": _tool_args_post_init,
            "to_tool_config": _tool_args_to_tool_config,
        },
    )

    return cls, field_mapping


def _tool_args_post_init(self: Any) -> None:
    """Validate tool arguments."""
    if self.tools:
        available = get_available_tools()
        for tool in self.tools:
            if tool.lower() not in available:
                raise ValueError(f"Unknown tool: {tool}. Available tools: {available}")

    if self.tool_parser not in get_available_parsers():
        raise ValueError(f"Unknown parser: {self.tool_parser}. Available: {get_available_parsers()}")

    if self.tool_tag_names is not None:
        if not self.tools:
            raise ValueError("--tool_tag_names requires --tools to be specified")
        if len(self.tool_tag_names) != len(self.tools):
            raise ValueError(
                f"--tool_tag_names must have same length as --tools. "
                f"Got {len(self.tool_tag_names)} for {len(self.tools)} tools."
            )


def _tool_args_to_tool_config(self: Any) -> "ToolConfig":
    """Convert flat ToolArgs to internal ToolConfig structure."""
    # Build kwargs for each config
    config_kwargs: dict[str, dict[str, Any]] = {attr: {} for attr in TOOL_CONFIG_REGISTRY}

    for cli_name, (config_attr, field_name) in self._field_mapping.items():
        config_kwargs[config_attr][field_name] = getattr(self, cli_name)

    # Instantiate configs
    tool_configs = {attr: cls(**config_kwargs[attr]) for attr, cls in TOOL_CONFIG_REGISTRY.items()}

    return ToolConfig(
        tools=self.tools,
        max_tool_calls=self.max_tool_calls,
        mask_tool_use=self.mask_tool_use,
        parser=self.tool_parser,
        tool_tag_names=self.tool_tag_names,
        **tool_configs,
    )


# Build ToolArgs at module load time
ToolArgs, _CLI_TO_CONFIG_MAPPING = _build_tool_args()

# Type hint for IDE (actual class is dynamic)
ToolArgs: type  # noqa: F811


# =============================================================================
# ToolConfig
# =============================================================================


@dataclass
class ToolConfig:
    """Internal structured tool configuration."""

    tools: list[str] | None = None
    max_tool_calls: int = 5
    mask_tool_use: bool = True
    parser: str = "legacy"
    tool_tag_names: list[str] | None = None

    python: PythonCodeToolConfig = field(default_factory=PythonCodeToolConfig)
    serper_search: SerperSearchToolConfig = field(default_factory=SerperSearchToolConfig)
    massive_ds_search: SearchToolConfig = field(default_factory=SearchToolConfig)
    s2_search: S2SearchToolConfig = field(default_factory=S2SearchToolConfig)
    you_search: YouSearchToolConfig = field(default_factory=YouSearchToolConfig)
    mcp: DrAgentMCPToolConfig = field(default_factory=DrAgentMCPToolConfig)


# =============================================================================
# Tool Setup
# =============================================================================


@dataclass
class ToolSetup:
    """Result of setting up tools."""

    tools: dict[str, Tool]
    parser: ToolParser | None
    stop_strings: list[str]


def build_tools_from_config(config: ToolConfig, vllm_tool_parser=None, vllm_output_formatter=None) -> ToolSetup:
    """Build tools and parser from ToolConfig."""
    if not config.tools:
        return ToolSetup(tools={}, parser=None, stop_strings=[])

    tools: dict[str, Tool] = {}
    tool_list: list[Tool] = []
    mcp_tools: list[Tool] = []

    for i, tool_name in enumerate(config.tools):
        tool_name_lower = tool_name.lower()
        config_attr, tool_cls, is_mcp_subtool = TOOL_REGISTRY[tool_name_lower]
        tool_config = getattr(config, config_attr)

        if config.tool_tag_names and hasattr(tool_config, "tag_name"):
            tool_config.tag_name = config.tool_tag_names[i]

        if is_mcp_subtool:
            tool = tool_cls.from_config(tool_config, tool_name_override=tool_name_lower)
        else:
            tool = tool_cls.from_config(tool_config)

        tools[tool.tool_function_name] = tool
        tool_list.append(tool)

        if isinstance(tool, DrAgentMCPTool):
            mcp_tools.append(tool)

    logger.info(f"Configured {len(tools)} tool(s): {list(tools.keys())}")

    stop_strings: list[str] = []

    if config.parser == "legacy":
        parser = OpenInstructLegacyToolParser(tool_list=tool_list)
        stop_strings = parser.stop_sequences()
    elif config.parser == "vllm":
        if vllm_tool_parser is None or vllm_output_formatter is None:
            raise ValueError("parser='vllm' requires vllm_tool_parser and vllm_output_formatter")
        parser = VllmToolParser(tool_parser=vllm_tool_parser, output_formatter=vllm_output_formatter)
        stop_strings = parser.stop_sequences()
    elif config.parser == "dr_tulu":
        assert len(mcp_tools) == 1 and len(tool_list) == 1, "DR Tulu only uses the MCP tool"
        parser = DRTuluToolParser(mcp_tool_list=mcp_tools)
        stop_strings = parser.stop_sequences()
    else:
        parser = None

    for tool in tool_list:
        if isinstance(tool, DrAgentMCPTool):
            stop_strings.extend(tool.get_stop_strings())

    return ToolSetup(tools=tools, parser=parser, stop_strings=list(set(stop_strings)))
