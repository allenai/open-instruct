"""
Tool configuration system for composable tool arguments.

Each tool defines its own Config dataclass (in tools.py) with fields
that can be configured via JSON config dicts on the CLI.

This module provides:
- ToolArgs: flat dataclass for CLI argument parsing (HfArgumentParser compatible)
- ToolConfig: internal structured config used by build_tools_from_config

CLI Usage:
    --tools mcp python --tool_configs '{"tool_names": "snippet_search"}' '{"api_endpoint": "..."}'

    The tool_configs list corresponds 1:1 with the tools list. Use {} for defaults.
"""

import json
import logging
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any

from open_instruct.tools.parsers import (
    DRTuluToolParser,
    OpenInstructLegacyToolParser,
    ToolParser,
    get_vllm_parser_mapping,
)
from open_instruct.tools.proxy import ToolProxy, create_tool_actor_from_config
from open_instruct.tools.tools import (
    DrAgentMCPToolConfig,
    GenericMCPToolConfig,
    MassiveDSSearchToolConfig,
    PythonCodeToolConfig,
    S2SearchToolConfig,
    SerperSearchToolConfig,
)
from open_instruct.tools.utils import BaseToolConfig, Tool

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Registry
# =============================================================================

# Maps tool name -> config_cls
# The config class must have `tool_class` as a ClassVar pointing to the Tool class.
# The dict key is used as both the tool name and the config attribute name.
#
# To add a new tool:
#   1. Create YourTool and YourToolConfig in tools.py
#      - YourToolConfig should inherit from BaseToolConfig
#      - Set tool_class as a ClassVar on the config
#   2. Add an entry here: "mytool": MyToolConfig
#
# Example:
#     # In tools.py
#     @dataclass
#     class MyToolConfig(BaseToolConfig):
#         tool_class: ClassVar[type[Tool]] = MyTool
#         some_option: int = 10
#         # override_name inherited from BaseToolConfig
#
#     # In config.py, add to TOOL_REGISTRY:
#     "mytool": MyToolConfig,
#
#     # Now you can use:
#     #   --tools mytool --tool_configs '{"some_option": 20}'

TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    "python": PythonCodeToolConfig,
    "serper_search": SerperSearchToolConfig,
    "massive_ds_search": MassiveDSSearchToolConfig,
    "s2_search": S2SearchToolConfig,
    "mcp": DrAgentMCPToolConfig,
    "generic_mcp": GenericMCPToolConfig,
}


# Built-in parsers that don't use vLLM
_BUILTIN_PARSERS = ("legacy", "dr_tulu")


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return list(_BUILTIN_PARSERS) + list(get_vllm_parser_mapping().keys())


def get_tool_definitions_from_config(config: "ToolConfig") -> list[dict[str, Any]]:
    """Get OpenAI-format tool definitions from ToolConfig.

    This instantiates each tool to get its definitions, which allows tools
    to dynamically discover their schemas (e.g., GenericMCPTool discovers
    tools from an MCP server).

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

        tool_config = config.get_tool_config(tool_name_lower)

        # Apply override name if specified
        if config.tool_override_names and i < len(config.tool_override_names):
            tool_config.override_name = config.tool_override_names[i]

        try:
            tool_instance = tool_config.build()
            definitions.extend(tool_instance.get_openai_tool_definitions())
        except Exception as e:
            logger.warning(f"Failed to instantiate tool {tool_name} for definitions: {e}")

    return definitions


def _build_tool_args() -> type:
    """
    Build ToolArgs dataclass for CLI parsing.

    Returns:
        The ToolArgs class with tool_configs list field.
    """
    all_fields: list[tuple[str, type, Any]] = [
        ("tools", list[str] | None, field(default=None)),
        ("tool_configs", list[str] | None, field(default=None)),
        ("max_tool_calls", int, field(default=5)),
        ("mask_tool_use", bool, field(default=True)),
        ("tool_parser", str, field(default="legacy")),
        ("tool_override_names", list[str] | None, field(default=None)),
    ]

    # Create class
    cls = make_dataclass(
        "ToolArgs",
        all_fields,
        namespace={
            "__doc__": """
Tool arguments for CLI parsing with HfArgumentParser.

Usage:
    --tools mcp python --tool_configs '{"tool_names": "snippet_search"}' '{"api_endpoint": "..."}'

The tool_configs list corresponds 1:1 with the tools list. Use '{}' for defaults.

Use to_tool_config() to convert to the internal ToolConfig structure.
""",
            "__post_init__": _tool_args_post_init,
            "to_tool_config": _tool_args_to_tool_config,
        },
    )

    return cls


def _tool_args_post_init(self: Any) -> None:
    """Validate tool arguments and parse JSON configs."""
    # Validate tools
    if self.tools:
        available = get_available_tools()
        for tool in self.tools:
            if tool.lower() not in available:
                raise ValueError(f"Unknown tool: {tool}. Available tools: {available}")

    # Validate tool_configs length matches tools
    if self.tool_configs is not None:
        if not self.tools:
            raise ValueError("--tool_configs requires --tools to be specified")
        if len(self.tool_configs) != len(self.tools):
            raise ValueError(
                f"--tool_configs must have same length as --tools. "
                f"Got {len(self.tool_configs)} configs for {len(self.tools)} tools."
            )

    # Parse and validate JSON configs
    parsed_configs: list[dict[str, Any]] = []
    if self.tools and self.tool_configs:
        for i, (tool_name, config_str) in enumerate(zip(self.tools, self.tool_configs)):
            tool_name_lower = tool_name.lower()
            config_cls = TOOL_REGISTRY[tool_name_lower]

            try:
                config_dict = json.loads(config_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool_configs[{i}] for '{tool_name}': {e}") from e

            if not isinstance(config_dict, dict):
                raise ValueError(
                    f"tool_configs[{i}] for '{tool_name}' must be a JSON object, got {type(config_dict).__name__}"
                )

            # Validate keys
            all_config_fields = {f.name for f in fields(config_cls)}
            for key in config_dict:
                if key not in all_config_fields:
                    raise ValueError(
                        f"Unknown key '{key}' in tool_configs[{i}] for '{tool_name}'. "
                        f"Valid keys: {list(all_config_fields)}"
                    )

            parsed_configs.append(config_dict)

    # Store parsed configs for use in to_tool_config()
    object.__setattr__(self, "_parsed_configs", parsed_configs)

    if self.tool_parser not in get_available_parsers():
        raise ValueError(f"Unknown parser: {self.tool_parser}. Available: {get_available_parsers()}")

    if self.tool_override_names is not None:
        if not self.tools:
            raise ValueError("--tool_override_names requires --tools to be specified")
        if len(self.tool_override_names) != len(self.tools):
            raise ValueError(
                f"--tool_override_names must have same length as --tools. "
                f"Got {len(self.tool_override_names)} for {len(self.tools)} tools."
            )


def _tool_args_to_tool_config(self: Any) -> "ToolConfig":
    """Convert ToolArgs to internal ToolConfig structure."""
    parsed_configs = getattr(self, "_parsed_configs", [])

    # Build tool configs dict - start with defaults
    tool_configs_dict = {name: config_cls() for name, config_cls in TOOL_REGISTRY.items()}

    # Override with parsed configs for specified tools
    if self.tools:
        for i, tool_name in enumerate(self.tools):
            tool_name_lower = tool_name.lower()
            config_cls = TOOL_REGISTRY[tool_name_lower]
            config_kwargs = parsed_configs[i] if i < len(parsed_configs) else {}
            tool_configs_dict[tool_name_lower] = config_cls(**config_kwargs)

    return ToolConfig(
        tools=self.tools,
        max_tool_calls=self.max_tool_calls,
        mask_tool_use=self.mask_tool_use,
        parser=self.tool_parser,
        tool_override_names=self.tool_override_names,
        tool_configs=tool_configs_dict,
    )


# Build ToolArgs at module load time
ToolArgs = _build_tool_args()


def _default_tool_configs() -> dict[str, BaseToolConfig]:
    """Create default configs for all registered tools."""
    return {name: config_cls() for name, config_cls in TOOL_REGISTRY.items()}


@dataclass
class ToolConfig:
    """Internal structured tool configuration."""

    tools: list[str] | None = None
    max_tool_calls: int = 5
    mask_tool_use: bool = True
    parser: str = "legacy"
    tool_override_names: list[str] | None = None
    tool_configs: dict[str, BaseToolConfig] = field(default_factory=_default_tool_configs)

    def get_tool_config(self, tool_name: str) -> BaseToolConfig:
        """Get the config for a specific tool."""
        return self.tool_configs[tool_name.lower()]


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
        from open_instruct.tools.parsers import create_vllm_parser

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

    Each tool name returned by get_tool_names() is registered as a separate
    entry in the tools dict. For tools that expose multiple names (like
    GenericMCPTool), each name gets a bound proxy pointing to the same actor.

    Note: This function does NOT create the parser. Use create_tool_parser()
    to create the parser lazily when needed (e.g., inside a Ray actor where
    the tokenizer is available).

    Args:
        config: The tool configuration.

    Returns:
        Tuple of (tools dict, stop_strings list)
    """
    if not config.tools:
        return {}, []

    proxies: dict[str, ToolProxy] = {}
    proxy_list: list[ToolProxy] = []

    for i, tool_name in enumerate(config.tools):
        tool_name_lower = tool_name.lower()
        tool_config = config.get_tool_config(tool_name_lower)

        if config.tool_override_names and hasattr(tool_config, "override_name"):
            tool_config.override_name = config.tool_override_names[i]

        # Step 1: Create ToolActor from config (config.build() called inside Ray actor)
        actor = create_tool_actor_from_config(config=tool_config)

        # Step 2: Wrap ToolActor with ToolProxy
        proxy = ToolProxy.from_actor(actor)

        # Step 3: Register a proxy for each tool name
        tool_names = proxy.get_tool_names()
        for name in tool_names:
            # If the tool exposes multiple names, bind each to route correctly
            if len(tool_names) > 1:
                proxies[name] = proxy.bind_to_tool(name)
            else:
                proxies[name] = proxy

        proxy_list.append(proxy)

    logger.info(f"Configured {len(proxy_list)} tool instance(s), {len(proxies)} tool name(s): {list(proxies.keys())}")

    # Get stop strings from proxies (fetched from actors)
    stop_strings: list[str] = []
    for proxy in proxy_list:
        stop_strings.extend(proxy.get_stop_strings())

    return proxies, list(set(stop_strings))
