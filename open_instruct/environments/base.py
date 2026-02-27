"""Base classes for RL environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from open_instruct.data_types import ToolCallStats


class EnvCall(Action):
    """Parsed action call from model output."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class StepResult(Observation):
    """Result from an environment step."""

    result: str = ""
    reward: float = 0.0


@dataclass
class RolloutState:
    """Accumulated state from a rollout (tools and/or environments)."""

    rewards: list[float] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    tool_error: str = ""
    tool_runtime: float = 0.0
    timeout: bool = False
    tool_call_stats: list[ToolCallStats] = field(default_factory=list)


class RLEnvironment(ABC):
    """Abstract base class for RL environments and tools.

    Subclass directly for stateful environments (games, sandboxes, etc.).
    Subclass Tool for stateless tool-based environments (code exec, web search, etc.).
    """

    config_name: str = ""

    async def setup(self) -> None:
        """Called once at start of training for resource initialization."""
        return

    def get_metrics(self) -> dict[str, float]:
        """Return custom metrics."""
        return {}

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """Return tool definitions in OpenAI format for prompt injection."""
        return []

    @abstractmethod
    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
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


@dataclass
class BaseEnvConfig:
    """Base configuration class for tools and environments.

    Subclasses pair with a tool/env class via tool_class and hold
    construction-time config (API endpoints, game parameters, etc.).
    """

    tool_class: ClassVar[type[RLEnvironment]]
    """The RLEnvironment subclass this config creates."""
