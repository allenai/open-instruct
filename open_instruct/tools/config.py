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

from open_instruct.tools.base import DRTuluToolParser, OpenInstructLegacyToolParser, Tool, ToolParser
from open_instruct.tools.tools import (
    MCP_TOOL_REGISTRY,
    DrAgentMCPTool,
    DrAgentMCPToolConfig,
    MassiveDSSearchTool,
    MassiveDSSearchToolConfig,
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Tool Registry
# =============================================================================


@dataclass(frozen=True)
class ToolEntry:
    """Metadata for a registered tool.

    This is the single source of truth for tool registration. Each entry defines:
    - The tool class and its config class
    - The config attribute name (used in ToolConfig dataclass)
    - Whether this is an MCP subtool (uses DrAgentMCPTool with tool_name_override)

    To add a new tool:
    1. Create YourTool and YourToolConfig in tools.py
    2. Add a ToolEntry to TOOL_REGISTRY below
    3. Add the config attribute to ToolConfig dataclass
    """

    tool_cls: type[Tool]
    config_cls: type
    config_attr: str  # Attribute name on ToolConfig (e.g., "python", "serper_search")
    is_mcp_subtool: bool = False  # If True, passes tool name to DrAgentMCPTool.from_config()


# Primary tool registry - maps canonical names to their entries
_TOOL_ENTRIES: dict[str, ToolEntry] = {
    "python": ToolEntry(PythonCodeTool, PythonCodeToolConfig, "python"),
    "serper_search": ToolEntry(SerperSearchTool, SerperSearchToolConfig, "serper_search"),
    "massive_ds_search": ToolEntry(MassiveDSSearchTool, MassiveDSSearchToolConfig, "massive_ds_search"),
    "s2_search": ToolEntry(S2SearchTool, S2SearchToolConfig, "s2_search"),
    "mcp": ToolEntry(DrAgentMCPTool, DrAgentMCPToolConfig, "mcp"),
}

# Aliases for convenience (e.g., --tools code instead of --tools python)
_TOOL_ALIASES: dict[str, str] = {"code": "python", "search": "serper_search"}

# Build the full registry including aliases and MCP subtools
TOOL_REGISTRY: dict[str, ToolEntry] = {}

# Add primary entries
for name, entry in _TOOL_ENTRIES.items():
    TOOL_REGISTRY[name] = entry

# Add aliases (point to the same entry)
for alias, canonical in _TOOL_ALIASES.items():
    TOOL_REGISTRY[alias] = _TOOL_ENTRIES[canonical]

# Add MCP sub-tools (snippet_search, google_search, etc.)
# These use the MCP entry but with is_mcp_subtool=True
_mcp_entry = _TOOL_ENTRIES["mcp"]
for mcp_name in MCP_TOOL_REGISTRY:
    TOOL_REGISTRY[mcp_name] = ToolEntry(
        tool_cls=_mcp_entry.tool_cls,
        config_cls=_mcp_entry.config_cls,
        config_attr=_mcp_entry.config_attr,
        is_mcp_subtool=True,
    )

# Derive config registry from tool entries (for CLI generation)
# Maps config_attr -> config_cls
TOOL_CONFIG_REGISTRY: dict[str, type] = {entry.config_attr: entry.config_cls for entry in _TOOL_ENTRIES.values()}

# Fields to exclude from CLI exposure (common across all configs)
EXCLUDED_FIELDS = {"tag_name", "cli_prefix"}

PARSER_TYPES = Literal["legacy", "vllm_hermes", "vllm_llama3", "vllm_qwen3_xml", "vllm_qwen3_coder", "dr_tulu"]

# Maps parser names to vllm_parsers factory function names
VLLM_PARSER_MAPPING: dict[str, str] = {
    "vllm_hermes": "hermes",
    "vllm_llama3": "llama3_json",
    "vllm_qwen3_xml": "qwen3_xml",
    "vllm_qwen3_coder": "qwen3_coder",
}


def get_parser_stop_sequences(parser_name: str) -> list[str]:
    """Get stop sequences for a parser.

    For vLLM parsers, these are the sequences that indicate a tool call is complete.
    For legacy/dr_tulu parsers, stop sequences come from the tools themselves.

    Args:
        parser_name: The parser type name.

    Returns:
        List of stop sequences for this parser.
    """
    if parser_name in VLLM_PARSER_MAPPING:
        from open_instruct.tools.vllm_parsers import VLLM_PARSERS

        vllm_name = VLLM_PARSER_MAPPING[parser_name]
        if vllm_name in VLLM_PARSERS:
            return list(VLLM_PARSERS[vllm_name].stop_sequences)
    return []


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "vllm_hermes", "vllm_llama3", "vllm_qwen3_xml", "vllm_qwen3_coder", "dr_tulu"]


def get_tool_definitions_from_config(config: "ToolConfig") -> list[dict[str, Any]]:
    """Get OpenAI-format tool definitions from ToolConfig without full tool instantiation.

    This is useful for passing tool definitions to apply_chat_template before
    the full tools are instantiated (which may require Ray actors, API endpoints, etc).

    Args:
        config: The tool configuration.

    Returns:
        List of tool definitions in OpenAI function calling format.
    """
    if not config.tools:
        return []

    definitions = []
    for i, tool_name in enumerate(config.tools):
        tool_name_lower = tool_name.lower()
        if tool_name_lower not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {get_available_tools()}")

        entry = TOOL_REGISTRY[tool_name_lower]

        # Get tag name from config or use default
        if config.tool_tag_names and i < len(config.tool_tag_names):
            tag_name = config.tool_tag_names[i]
        else:
            tag_name = getattr(entry.tool_cls, "_default_tool_function_name", tool_name_lower)

        # Get description and parameters from class attributes
        description = getattr(entry.tool_cls, "_default_tool_description", "")
        parameters = getattr(
            entry.tool_cls, "_default_tool_parameters", {"type": "object", "properties": {}, "required": []}
        )

        definitions.append(
            {"type": "function", "function": {"name": tag_name, "description": description, "parameters": parameters}}
        )

    return definitions


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
    massive_ds_search: MassiveDSSearchToolConfig = field(default_factory=MassiveDSSearchToolConfig)
    s2_search: S2SearchToolConfig = field(default_factory=S2SearchToolConfig)
    mcp: DrAgentMCPToolConfig = field(default_factory=DrAgentMCPToolConfig)


# =============================================================================
# Tool Setup
# =============================================================================


def create_tool_parser(parser_name: str, tokenizer=None, tools: dict[str, Tool] | None = None) -> ToolParser | None:
    """Create a tool parser by name.

    This function creates the appropriate parser based on the parser name.
    It's designed to be called lazily (e.g., inside a Ray actor) to avoid
    serialization issues with parsers that contain non-serializable components.

    Args:
        parser_name: The parser type (e.g., "legacy", "vllm_qwen3_xml", "dr_tulu").
        tokenizer: Required for vllm_* parsers. The tokenizer for the model.
        tools: Dict of tool name -> Tool. Required for "legacy" and "dr_tulu" parsers.
               For vllm_* parsers, used to extract tool definitions.

    Returns:
        The created ToolParser, or None if parser_name is None/empty.
    """
    if not parser_name:
        return None

    if parser_name == "legacy":
        if not tools:
            raise ValueError("parser='legacy' requires tools to be provided")
        return OpenInstructLegacyToolParser(tool_list=list(tools.values()))

    elif parser_name in VLLM_PARSER_MAPPING:
        if tokenizer is None:
            raise ValueError(f"parser='{parser_name}' requires a tokenizer")
        from open_instruct.tools.vllm_parsers import create_vllm_parser

        # Extract tool definitions for vLLM parsers
        tool_definitions = None
        if tools:
            tool_definitions = [tool.get_openai_tool_definition() for tool in tools.values()]

        vllm_parser_name = VLLM_PARSER_MAPPING[parser_name]
        return create_vllm_parser(vllm_parser_name, tokenizer, tool_definitions=tool_definitions)

    elif parser_name == "dr_tulu":
        if not tools:
            raise ValueError("parser='dr_tulu' requires tools to be provided")
        # DR Tulu parser requires MCP tools - check that at least one MCP tool is configured
        if "mcp" not in tools:
            raise ValueError(
                "parser='dr_tulu' requires the 'mcp' tool to be configured. "
                "Add '--tools mcp' to your command line arguments."
            )
        return DRTuluToolParser(mcp_tool_list=list(tools.values()))

    else:
        logger.warning(f"Unknown tool parser: {parser_name}")
        return None


def build_tools_from_config(config: ToolConfig) -> tuple[dict[str, Tool], list[str]]:
    """Build tools from ToolConfig.

    All tools are created as ToolProxy instances that instantiate the actual
    tools inside Ray actors. This provides a uniform pattern and avoids
    serialization issues with tools that have heavy dependencies.

    Note: This function does NOT create the parser. Use create_tool_parser()
    to create the parser lazily when needed (e.g., inside a Ray actor where
    the tokenizer is available).

    Args:
        config: The tool configuration.

    Returns:
        Tuple of (tools dict, stop_strings list)
    """
    # Import here to avoid circular imports
    from open_instruct.tools.proxy import ToolProxy, create_tool_actor_from_config

    if not config.tools:
        return {}, []

    proxies: dict[str, ToolProxy] = {}
    proxy_list: list[ToolProxy] = []

    for i, tool_name in enumerate(config.tools):
        tool_name_lower = tool_name.lower()
        entry = TOOL_REGISTRY[tool_name_lower]
        tool_config = getattr(config, entry.config_attr)

        if config.tool_tag_names and hasattr(tool_config, "tag_name"):
            tool_config.tag_name = config.tool_tag_names[i]

        # Derive class_path from tool_cls
        class_path = f"{entry.tool_cls.__module__}:{entry.tool_cls.__name__}"

        # For MCP sub-tools (snippet_search, google_search, etc.), pass tool_name_override
        tool_name_override = tool_name_lower if entry.is_mcp_subtool else None

        # Step 1: Create ToolActor from config (tool instantiated inside Ray actor)
        actor = create_tool_actor_from_config(
            class_path=class_path, config=tool_config, tool_name_override=tool_name_override
        )

        # Step 2: Wrap ToolActor with ToolProxy
        proxy = ToolProxy.from_actor(actor)

        proxies[proxy.tool_function_name] = proxy
        proxy_list.append(proxy)

    logger.info(f"Configured {len(proxies)} tool(s): {list(proxies.keys())}")

    # Get stop strings from proxies (fetched from actors)
    stop_strings: list[str] = []
    for proxy in proxy_list:
        stop_strings.extend(proxy.get_stop_strings())

    return proxies, list(set(stop_strings))
