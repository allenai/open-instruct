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
    response_role: str = "tool"

    def get_response_role(self) -> str:
        return self.response_role

    def get_is_text_env(self) -> bool:
        return False

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
    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
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


class TextRLEnvironment(RLEnvironment):
    """Base class for text-based RL environments.

    Unlike tool-based environments where parsed tool calls are dispatched,
    text environments receive the model's entire generation as a plain string.
    The environment's response is injected back into the conversation with
    the role specified by ``response_role`` (default ``"user"``).

    Subclasses implement ``text_step`` instead of ``step``.

    The dispatch pipeline injects a shadow ``EnvCall`` with the model's full
    output stored under ``args["text"]``. The ``step`` method extracts this
    and forwards it to ``text_step``.
    """

    response_role: str = "user"

    @abstractmethod
    async def text_step(self, text: str) -> StepResult:
        """Process the model's full output text and return an observation."""
        pass

    @abstractmethod
    async def _reset(self, **kwargs: Any) -> StepResult:
        """Subclass hook: initialize episode and return the initial observation."""
        pass

    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        """Initialize episode. Always returns empty tool list for text envs."""
        result = await self._reset(**kwargs)
        return result, []

    async def step(self, call: EnvCall) -> StepResult:
        """Extract ``args["text"]`` from the shadow EnvCall and forward to text_step."""
        return await self.text_step(call.args["text"])

    def get_is_text_env(self) -> bool:
        return True


@dataclass
class BaseEnvConfig:
    """Base configuration class for tools and environments.

    Subclasses pair with a tool/env class via tool_class and hold
    construction-time config (API endpoints, game parameters, etc.).
    """

    tool_class: ClassVar[type[RLEnvironment]]
    """The RLEnvironment subclass this config creates."""
