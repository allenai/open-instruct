"""Base classes for RL environments."""

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from open_instruct.tools.utils import ToolCall
from open_instruct.executable import Executable, ExecutableOutput


@dataclass
class StepResult:
    """Result from an environment step."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """Accumulated state from an environment rollout."""

    episode_id: str | None = None
    rewards: list[float] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def final_reward(self) -> float:
        return self.rewards[-1] if self.rewards else 0.0

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)


class RLEnvironment(Executable):
    """Abstract base class for RL environments (use as Ray actors via ray.remote).

    Subclass this to implement your own environment, defining reset/state/step.
    """

    async def _execute(self, _name_: str = "", _id_: str = "", **kwargs) -> ExecutableOutput:
        """Delegates to step(), wrapping the result as ExecutableOutput."""
        start = time.perf_counter()
        tc = ToolCall(id=_id_, name=_name_, args=kwargs)
        try:
            result = await self.step(tc)
            return ExecutableOutput(
                output=result.observation or "",
                called=True,
                error="",
                timeout=False,
                runtime=time.perf_counter() - start,
                reward=result.reward,
                done=result.done,
                info=result.info,
            )
        except Exception as e:
            return ExecutableOutput(
                output=f"Error: {e}",
                called=True,
                error=str(e),
                timeout=False,
                runtime=time.perf_counter() - start,
                reward=0.0,
                done=False,
                info={},
            )

    @abstractmethod
    async def reset(self, task_id: str | None = None, **kwargs) -> tuple[StepResult, list[dict]]:
        """Initialize episode. Returns (initial observation, tool definitions)."""
        pass

    @abstractmethod
    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute action, return observation, reward, done."""
        pass

    @abstractmethod
    def state(self) -> EnvironmentState:
        """Return current episode state."""
        pass


ENV_REGISTRY: dict[str, type[RLEnvironment]] = {}


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: type[RLEnvironment]) -> type[RLEnvironment]:
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment '{name}' already registered")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


def get_env_class(env_name: str) -> type[RLEnvironment]:
    """Get environment class by registry name."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Environment '{env_name}' not found. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[env_name]
