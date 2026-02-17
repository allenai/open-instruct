"""Base classes for RL environments."""

import importlib
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from open_instruct.tools.utils import Tool, ToolCall, ToolOutput


@dataclass
class StepResult:
    """Result from environment reset or step."""

    observation: str
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    tools: list[dict] | None = None


ENV_REGISTRY: dict[str, type["RLEnvironment"]] = {}


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: type["RLEnvironment"]) -> type["RLEnvironment"]:
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment '{name}' already registered")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


class RLEnvironment(Tool):
    """Abstract base class for RL environments (use as Ray actors via ray.remote).

    Extends Tool so that environments and regular tools share a common base.
    Environments use reset()/step() instead of _execute().
    """

    # Role for model output in conversation (used when no tool parser)
    response_role: str = "assistant"

    async def _execute(self, **kwargs) -> ToolOutput:
        """Not used by environments — they use step() via safe_execute()."""
        raise NotImplementedError("RLEnvironment uses step() via safe_execute()")

    @abstractmethod
    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        """Initialize episode, return observation and tools."""
        pass

    @abstractmethod
    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute action, return observation, reward, done."""
        pass

    async def safe_execute(self, _name_: str = "", _id_: str | None = None, **kwargs) -> ToolOutput:
        """Unified interface matching regular tools.

        Wraps step() and returns a ToolOutput so that callers can use
        the same dispatch path for both environments and regular tools.
        """
        start = time.perf_counter()
        tc = ToolCall(name=_name_, args=kwargs, id=_id_)
        try:
            result = await self.step(tc)
            return ToolOutput(
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
            return ToolOutput(
                output=f"Error: {e}",
                called=True,
                error=str(e),
                timeout=False,
                runtime=time.perf_counter() - start,
                reward=0.0,
                done=False,
                info={},
            )


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
