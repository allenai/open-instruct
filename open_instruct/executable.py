"""Shared data types for tools and environments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Parsed tool call from model output."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ExecutableOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float
    reward: float | None = None
    done: bool = False
    info: dict = field(default_factory=dict)
