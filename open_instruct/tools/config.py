"""
Tool configuration system for composable tool arguments.

Each tool defines its own Config dataclass (in tools.py) with fields
that can be configured via JSON config dicts on the CLI.

This module provides:
- ToolArgs: flat dataclass for CLI argument parsing (HfArgumentParser compatible)
- ToolConfig: internal structured config used by build_tools_from_config

CLI Usage:
    --tools python serper_search --tool_configs '{"api_endpoint": "..."}' '{}'

    The tool_configs list corresponds 1:1 with the tools list. Use {} for defaults.
"""

import json
import logging
from dataclasses import dataclass, field, fields
from typing import Any

import ray

from open_instruct.tools.parsers import OpenInstructLegacyToolParser, ToolParser, get_available_parsers
from open_instruct.tools.proxy import create_tool_actor_from_config
from open_instruct.tools.tools import PythonCodeToolConfig, S2SearchToolConfig, SerperSearchToolConfig
from open_instruct.tools.utils import BaseToolConfig

logger = logging.getLogger(__name__)


TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    "python": PythonCodeToolConfig,
    "serper_search": SerperSearchToolConfig,
    "s2_search": S2SearchToolConfig,
}


def get_available_tools() -> list[str]:
    """Return list of available tool names."""
    return list(TOOL_REGISTRY.keys())


@dataclass
class ToolArgs:
    """
    Tool arguments for CLI parsing with HfArgumentParser.

    Usage:
        --tools python serper_search --tool_configs '{"api_endpoint": "..."}' '{}'

    The tool_configs list corresponds 1:1 with the tools list. Use '{}' for defaults.
    Override names are set via the 'override_name' field in each tool's config JSON.

    Use to_tool_config() to convert to the internal ToolConfig structure.
    """

    tools: list[str] | None = None
    tool_configs: list[str] | None = None
    max_tool_calls: int = 5
    tool_parser: str = "legacy"
    pass_tools_to_chat_template: bool = True
    """Whether to pass tool definitions to the chat template (for models with native function calling)."""

    _parsed_configs: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.tools:
            available = get_available_tools()
            for tool in self.tools:
                if tool.lower() not in available:
                    raise ValueError(f"Unknown tool: {tool}. Available tools: {available}")

        if self.tool_configs is not None:
            if not self.tools:
                raise ValueError("--tool_configs requires --tools to be specified")
            if len(self.tool_configs) != len(self.tools):
                raise ValueError(
                    f"--tool_configs must have same length as --tools. "
                    f"Got {len(self.tool_configs)} configs for {len(self.tools)} tools."
                )

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

                all_config_fields = {f.name for f in fields(config_cls)}
                for key in config_dict:
                    if key not in all_config_fields:
                        raise ValueError(
                            f"Unknown key '{key}' in tool_configs[{i}] for '{tool_name}'. "
                            f"Valid keys: {list(all_config_fields)}"
                        )

                parsed_configs.append(config_dict)

        object.__setattr__(self, "_parsed_configs", parsed_configs)

        if self.tool_parser not in get_available_parsers():
            raise ValueError(f"Unknown parser: {self.tool_parser}. Available: {get_available_parsers()}")

    def to_tool_config(self) -> "ToolConfig":
        """Convert ToolArgs to internal ToolConfig structure."""
        parsed_configs = self._parsed_configs
        tool_configs_list: list[BaseToolConfig] = []
        if self.tools:
            for i, tool_name in enumerate(self.tools):
                tool_name_lower = tool_name.lower()
                config_cls = TOOL_REGISTRY[tool_name_lower]
                config_kwargs = parsed_configs[i] if i < len(parsed_configs) else {}
                tool_configs_list.append(config_cls(**config_kwargs))

        return ToolConfig(
            tools=self.tools,
            max_tool_calls=self.max_tool_calls,
            parser=self.tool_parser,
            tool_configs_list=tool_configs_list,
        )


@dataclass
class ToolConfig:
    """Master config for tools, containing the list of tool configs and higher-level configuration (e.g., what parser to use)."""

    tools: list[str] | None = None
    max_tool_calls: int = 5
    parser: str = "legacy"
    tool_configs_list: list[BaseToolConfig] = field(default_factory=list)

    def get_tool_config(self, index: int) -> BaseToolConfig:
        """Get the config for a tool by index."""
        return self.tool_configs_list[index]


def create_tool_parser(
    parser_name: str, tokenizer=None, tools: dict[str, ray.actor.ActorHandle] | None = None
) -> ToolParser | None:
    """Create a tool parser by name.

    This function creates the appropriate parser based on the parser name.
    It's designed to be called lazily (e.g., inside a Ray actor) to avoid
    serialization issues with parsers that contain non-serializable components.

    Args:
        parser_name: The parser type (currently only "legacy" is supported).
        tokenizer: Reserved for future parser types.
        tools: Dict of tool name -> ActorHandle. Required for "legacy" parser.

    Returns:
        The created ToolParser, or None if parser_name is None/empty.
    """
    if not parser_name:
        return None

    if parser_name == "legacy":
        if not tools:
            raise ValueError("parser='legacy' requires tools to be provided")
        return OpenInstructLegacyToolParser(tool_actors=list(tools.values()))

    else:
        logger.warning(f"Unknown tool parser: {parser_name}")
        return None


def build_tools_from_config(config: ToolConfig) -> tuple[dict[str, ray.actor.ActorHandle], list[str]]:
    """Build tools from ToolConfig.

    All tools are created as ToolActor instances inside Ray actors.
    This avoids serialization issues with tools that have heavy dependencies.

    Args:
        config: The tool configuration.

    Returns:
        Tuple of (tool_actors dict mapping name -> ActorHandle, stop_strings list)
    """
    if not config.tools:
        return {}, []

    tool_actors: dict[str, ray.actor.ActorHandle] = {}
    stop_strings: list[str] = []

    for i, _tool_name in enumerate(config.tools):
        tool_config = config.get_tool_config(i)

        actor = create_tool_actor_from_config(config=tool_config)
        name = ray.get(actor.get_tool_function_name.remote())
        if name in tool_actors:
            raise ValueError(
                f"Tool name collision: '{name}' is already registered. "
                f"Consider using override_name in tool_configs to use different names."
            )
        tool_actors[name] = actor
        stop_strings.extend(ray.get(actor.get_stop_strings.remote()))

    logger.info(f"Configured {len(tool_actors)} tool(s): {list(tool_actors.keys())}")

    return tool_actors, list(set(stop_strings))
