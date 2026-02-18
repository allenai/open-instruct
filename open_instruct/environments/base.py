"""Base classes for RL environments."""

from abc import ABC, abstractmethod
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class EnvCall(Action):
    """Parsed action call from model output."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class StepResult(Observation):
    """Result from an environment step."""

    result: str = ""
    reward: float = 0.0


class RLEnvironment(ABC):
    """Abstract base class for RL environments and tools.

    Subclass directly for stateful environments (games, sandboxes, etc.).
    Subclass Tool for stateless tool-based environments (code exec, web search, etc.).
    """

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
    async def reset(self, task_id: str | None = None, **kwargs) -> tuple[StepResult, list[dict]]:
        """Initialize episode. Returns (initial observation, tool definitions)."""
        pass

    @abstractmethod
    async def step(self, call: EnvCall) -> StepResult:
        """Execute action, return observation, reward, done."""
        pass

    @abstractmethod
    def state(self) -> State:
        """Return current episode state."""
        pass
