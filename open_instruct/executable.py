"""Base dispatch interface shared by tools and environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class ToolCall:
    """Parsed tool call from model output."""

    name: str
    args: dict[str, Any]
    id: str = ""


class Executable(ABC):
    """Base class for anything the vLLM actor can dispatch to (tools and environments)."""

    async def setup(self) -> None:
        """Called once at start of training for resource initialization."""
        return

    async def shutdown(self) -> None:
        """Called once at end of training for resource cleanup."""
        return

    def get_metrics(self) -> dict[str, float]:
        """Return custom metrics."""
        return {}

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """Return tool definitions in OpenAI format for prompt injection."""
        return []

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ExecutableOutput:
        """Execute. Must be implemented by subclasses."""
        raise NotImplementedError("_execute must be implemented by subclasses.")

    async def safe_execute(self, *args: Any, _name_: str = "", _id_: str = "", **kwargs: Any) -> ExecutableOutput:
        """Dispatch entry point. Calls _execute with the provided arguments."""
        return await self._execute(*args, _name_=_name_, _id_=_id_, **kwargs)

    async def __call__(self, *args: Any, **kwargs: Any) -> ExecutableOutput:
        """Alias for safe_execute, useful for inference scripts."""
        return await self.safe_execute(*args, **kwargs)
