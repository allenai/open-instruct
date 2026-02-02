"""Base classes for RL environments."""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import ray

from open_instruct.tools.utils import ToolCall


@dataclass
class StepResult:
    """Result from environment reset or step."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tools: list[dict] | None = None
    """Tool schemas (only returned from reset)."""


@dataclass
class EnvironmentState:
    """Accumulated state from an environment rollout."""

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


ENV_REGISTRY: dict[str, type["RLEnvironment"]] = {}


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: type["RLEnvironment"]) -> type["RLEnvironment"]:
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment '{name}' already registered")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


class RLEnvironment(ABC):
    """Abstract base class for RL environments (run as Ray actors via make_env_actor)."""

    use_tool_calls: bool = True
    response_role: str = "tool"
    max_steps: int = 50

    async def setup(self) -> None:
        """Called once at start of training for resource initialization."""
        pass

    @abstractmethod
    async def reset(self, task_id: str | None = None) -> StepResult:
        """Initialize episode, return observation and tools."""
        pass

    @abstractmethod
    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute action, return observation, reward, done."""
        pass

    def get_metrics(self) -> dict[str, float]:
        """Return custom metrics."""
        return {}

    async def close(self) -> None:
        """Cleanup resources for a single episode."""
        pass

    async def shutdown(self) -> None:
        """Called once at end of training for resource cleanup."""
        pass


def make_env_actor(env_class: type[RLEnvironment]) -> type:
    """Wrap an RLEnvironment class in a Ray actor."""

    @ray.remote
    class EnvironmentActor:
        def __init__(self, **kwargs):
            self._env = env_class(**kwargs)

        async def setup(self) -> None:
            await self._env.setup()

        async def reset(self, task_id: str | None = None) -> StepResult:
            return await self._env.reset(task_id)

        async def step(self, tool_call: ToolCall) -> StepResult:
            return await self._env.step(tool_call)

        def get_metrics(self) -> dict[str, float]:
            return self._env.get_metrics()

        async def close(self) -> None:
            await self._env.close()

        async def shutdown(self) -> None:
            await self._env.shutdown()

        @property
        def use_tool_calls(self) -> bool:
            return self._env.use_tool_calls

        @property
        def response_role(self) -> str:
            return self._env.response_role

        @property
        def max_steps(self) -> int:
            return self._env.max_steps

    EnvironmentActor.__name__ = f"{env_class.__name__}Actor"
    return EnvironmentActor


def get_env_class(env_name: str | None = None, env_class: str | None = None) -> type[RLEnvironment]:
    """Get environment class by registry name or import path."""
    if env_name is not None:
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Environment '{env_name}' not found. Available: {list(ENV_REGISTRY.keys())}")
        return ENV_REGISTRY[env_name]

    if env_class is not None:
        module_path, class_name = env_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not issubclass(cls, RLEnvironment):
            raise ValueError(f"{env_class} is not an RLEnvironment")
        return cls

    raise ValueError("Must provide env_name or env_class")
