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
from dataclasses import MISSING, field, fields, make_dataclass
from typing import Any

from open_instruct.tools.parsers import DRTuluToolParser, OpenInstructLegacyToolParser, ToolParser
from open_instruct.tools.tools import (
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
from open_instruct.tools.utils import Tool

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Registry
# =============================================================================

# Maps tool name -> (tool_cls, config_cls)
# The dict key is used as both the tool name and the config attribute name.
#
# To add a new tool:
#   1. Create YourTool and YourToolConfig in tools.py (with cli_prefix on config)
#   2. Add an entry here: "mytool": (MyTool, MyToolConfig)
#
# That's it! CLI args and internal config are generated automatically.
#
# Example:
#     # In tools.py
#     @dataclass
#     class MyToolConfig:
#         cli_prefix: ClassVar[str] = "mytool_"
#         some_option: int = 10
#
#     class MyTool(Tool):
#         @classmethod
#         def from_config(cls, config: MyToolConfig) -> "MyTool":
#             return cls(...)
#
#     # In config.py, add to TOOL_REGISTRY:
#     "mytool": (MyTool, MyToolConfig),
#
#     # Now you can use:
#     #   --tools mytool --mytool_some_option 20

TOOL_REGISTRY: dict[str, tuple[type[Tool], type]] = {
    "python": (PythonCodeTool, PythonCodeToolConfig),
    "serper_search": (SerperSearchTool, SerperSearchToolConfig),
    "massive_ds_search": (MassiveDSSearchTool, MassiveDSSearchToolConfig),
    "s2_search": (S2SearchTool, S2SearchToolConfig),
    "mcp": (DrAgentMCPTool, DrAgentMCPToolConfig),
}

# Fields to exclude from CLI exposure (common across all configs)
EXCLUDED_FIELDS = {"tag_name", "cli_prefix"}

# Built-in parsers that don't use vLLM
_BUILTIN_PARSERS = ("legacy", "dr_tulu")


def _get_vllm_parser_mapping() -> dict[str, str]:
    """Auto-generate vllm_X -> X mapping from VLLM_PARSERS.

    To add a new vLLM parser, just add it to VLLM_PARSERS in vllm_parsers.py.
    The CLI argument will automatically be available as --tool_parser vllm_{name}.
    """
    from open_instruct.tools.vllm_parsers import VLLM_PARSERS

    return {f"vllm_{name}": name for name in VLLM_PARSERS}


# Lazy-load to avoid circular import at module level
_VLLM_PARSER_MAPPING_CACHE: dict[str, str] | None = None


def get_vllm_parser_mapping() -> dict[str, str]:
    """Get the vLLM parser name mapping (lazy-loaded)."""
    global _VLLM_PARSER_MAPPING_CACHE
    if _VLLM_PARSER_MAPPING_CACHE is None:
        _VLLM_PARSER_MAPPING_CACHE = _get_vllm_parser_mapping()
    return _VLLM_PARSER_MAPPING_CACHE


def get_parser_stop_sequences(parser_name: str) -> list[str]:
    """Get stop sequences for a parser.

    For vLLM parsers, these are the sequences that indicate a tool call is complete.
    For legacy/dr_tulu parsers, stop sequences come from the tools themselves.

    Args:
        parser_name: The parser type name.

    Returns:
        List of stop sequences for this parser.
    """
    vllm_mapping = get_vllm_parser_mapping()
    if parser_name in vllm_mapping:
        from open_instruct.tools.vllm_parsers import VLLM_PARSERS

        vllm_name = vllm_mapping[parser_name]
        if vllm_name in VLLM_PARSERS:
            return list(VLLM_PARSERS[vllm_name].stop_sequences)
    return []


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return list(_BUILTIN_PARSERS) + list(get_vllm_parser_mapping().keys())


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

        tool_cls, _ = TOOL_REGISTRY[tool_name_lower]

        # Get tag name from config or use default
        if config.tool_tag_names and i < len(config.tool_tag_names):
            tag_name = config.tool_tag_names[i]
        else:
            tag_name = getattr(tool_cls, "_default_tool_function_name", tool_name_lower)

        # Get description and parameters from class attributes
        description = getattr(tool_cls, "_default_tool_description", "")
        parameters = getattr(
            tool_cls, "_default_tool_parameters", {"type": "object", "properties": {}, "required": []}
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

    # Add fields from each tool config (tool_name is config_attr)
    for config_attr, (_, config_cls) in TOOL_REGISTRY.items():
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
    # Build kwargs for each config (tool_name is config_attr)
    config_kwargs: dict[str, dict[str, Any]] = {name: {} for name in TOOL_REGISTRY}

    for cli_name, (config_attr, field_name) in self._field_mapping.items():
        config_kwargs[config_attr][field_name] = getattr(self, cli_name)

    # Instantiate configs
    tool_configs = {name: config_cls(**config_kwargs[name]) for name, (_, config_cls) in TOOL_REGISTRY.items()}

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
# ToolConfig (auto-generated)
# =============================================================================


def _build_tool_config() -> type:
    """Build ToolConfig dataclass from tool registry.

    This auto-generates the ToolConfig class with a field for each tool's config.
    """
    # Base fields
    base_fields: list[tuple[str, type, Any]] = [
        ("tools", list[str] | None, field(default=None)),
        ("max_tool_calls", int, field(default=5)),
        ("mask_tool_use", bool, field(default=True)),
        ("parser", str, field(default="legacy")),
        ("tool_tag_names", list[str] | None, field(default=None)),
    ]

    # Add a field for each tool config (tool_name is the config attr name)
    for tool_name, (_, config_cls) in TOOL_REGISTRY.items():
        base_fields.append((tool_name, config_cls, field(default_factory=config_cls)))

    return make_dataclass(
        "ToolConfig",
        base_fields,
        namespace={"__doc__": "Internal structured tool configuration. Auto-generated from TOOL_REGISTRY."},
    )


# Build ToolConfig at module load time
ToolConfig = _build_tool_config()

# Type hint for IDE (actual class is dynamic)
ToolConfig: type  # noqa: F811


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

    elif parser_name in get_vllm_parser_mapping():
        if tokenizer is None:
            raise ValueError(f"parser='{parser_name}' requires a tokenizer")
        from open_instruct.tools.vllm_parsers import create_vllm_parser

        # Extract tool definitions for vLLM parsers
        tool_definitions = None
        if tools:
            tool_definitions = [tool.get_openai_tool_definition() for tool in tools.values()]

        vllm_parser_name = get_vllm_parser_mapping()[parser_name]
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
        return DRTuluToolParser(tool_list=list(tools.values()))

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
        tool_cls, _ = TOOL_REGISTRY[tool_name_lower]
        tool_config = getattr(config, tool_name_lower)  # config attr = tool name

        if config.tool_tag_names and hasattr(tool_config, "tag_name"):
            tool_config.tag_name = config.tool_tag_names[i]

        # Derive class_path from tool_cls
        class_path = f"{tool_cls.__module__}:{tool_cls.__name__}"

        # Step 1: Create ToolActor from config (tool instantiated inside Ray actor)
        actor = create_tool_actor_from_config(class_path=class_path, config=tool_config)

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
