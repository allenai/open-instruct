"""
Base classes for RL environments.

Environments are Ray actors that manage stateful interactions with external systems.
They provide tool schemas dynamically via reset() and execute actions via step().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import ray


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""

    name: str  # Tool function name (e.g., "guess", "execute")
    args: dict[str, Any]  # Parsed arguments (e.g., {"word": "bread"})
    id: str | None = None  # Optional call ID for multi-tool responses


@dataclass
class ResetResult:
    """Result from environment reset - includes observation and available tools."""

    observation: str  # Initial observation/task description
    tools: list[dict]  # OpenAI-format tool schemas for this task
    info: dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class StepResult:
    """Result from environment step - includes observation, reward, and done flag."""

    observation: str  # Observation after taking action
    reward: float  # Reward for this step
    done: bool  # Whether episode is complete
    info: dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class EnvironmentState:
    """Tracks accumulated state from an RL environment during rollout."""

    rewards: list[float] = field(default_factory=list)
    step_count: int = 0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def final_reward(self) -> float:
        """Get the final reward (last reward in list, or 0.0 if empty)."""
        return self.rewards[-1] if self.rewards else 0.0

    @property
    def total_reward(self) -> float:
        """Get the sum of all rewards."""
        return sum(self.rewards)


# Environment registry for discovery
ENV_REGISTRY: dict[str, type["RLEnvironment"]] = {}


def register_env(name: str):
    """Decorator to register an environment class in the registry.

    Usage:
        @register_env("wordle")
        class WordleEnv(RLEnvironment):
            ...
    """

    def decorator(cls: type["RLEnvironment"]) -> type["RLEnvironment"]:
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment '{name}' already registered")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


class RLEnvironment(ABC):
    """
    Abstract base class for RL environments.

    Environments run as Ray actors - they are instantiated via make_env_actor()
    which wraps them in a Ray actor class. This avoids pickle issues with
    unpicklable state (Docker clients, HTTP sessions, E2B connections, etc.).

    Subclasses must implement:
        - reset(task_id) -> ResetResult
        - step(tool_call) -> StepResult

    Optional overrides:
        - get_metrics() -> dict[str, float]
        - close() -> None

    Class attributes:
        - use_tool_calls: If False, step() receives raw text instead of ToolCall
        - response_role: "tool" or "user" for conversation history
        - max_steps: Maximum steps before forced termination
    """

    # Class attributes (can be overridden in subclasses)
    use_tool_calls: bool = True  # If False, step() receives raw text as ToolCall(name="raw", args={"text": ...})
    response_role: str = "tool"  # "tool" or "user" for multi-turn emulation
    max_steps: int = 50  # Force done=True if exceeded

    @abstractmethod
    async def reset(self, task_id: str | None = None) -> ResetResult:
        """
        Initialize episode for the given task.

        Args:
            task_id: Optional task identifier for this episode.
                     If None, environment may sample a random task.

        Returns:
            ResetResult with initial observation and available tools.
        """
        pass

    @abstractmethod
    async def step(self, tool_call: ToolCall) -> StepResult:
        """
        Execute an action in the environment.

        Args:
            tool_call: The parsed tool call from model output.
                      If use_tool_calls=False, this will be
                      ToolCall(name="raw", args={"text": raw_output})

        Returns:
            StepResult with observation, reward, and done flag.
        """
        pass

    def get_metrics(self) -> dict[str, float]:
        """
        Return environment-specific metrics at episode end.

        Override in subclass to report custom metrics (e.g., guess count for Wordle).
        These are logged to wandb/tensorboard by the training loop.

        Returns:
            Dictionary of metric name -> value
        """
        return {}

    async def close(self) -> None:
        """
        Cleanup resources (optional override).

        Called when the environment actor is being shut down.
        Override to close connections, stop containers, etc.
        """
        pass


def make_env_actor(env_class: type[RLEnvironment]) -> type:
    """
    Create a Ray actor class from an RLEnvironment class.

    This wraps the environment class in a Ray actor, allowing it to run
    in a separate process without pickle issues.

    Usage:
        EnvActor = make_env_actor(MyEnv)
        actor = EnvActor.remote(**kwargs)
        result = await actor.reset.remote(task_id)

    Args:
        env_class: The RLEnvironment subclass to wrap

    Returns:
        A Ray actor class that wraps the environment
    """

    @ray.remote
    class EnvironmentActor:
        """Ray actor wrapper for RLEnvironment."""

        def __init__(self, **kwargs):
            self._env = env_class(**kwargs)

        async def reset(self, task_id: str | None = None) -> ResetResult:
            return await self._env.reset(task_id)

        async def step(self, tool_call: ToolCall) -> StepResult:
            return await self._env.step(tool_call)

        def get_metrics(self) -> dict[str, float]:
            return self._env.get_metrics()

        async def close(self) -> None:
            await self._env.close()

        # Expose class attributes
        @property
        def use_tool_calls(self) -> bool:
            return self._env.use_tool_calls

        @property
        def response_role(self) -> str:
            return self._env.response_role

        @property
        def max_steps(self) -> int:
            return self._env.max_steps

    # Copy class name for debugging
    EnvironmentActor.__name__ = f"{env_class.__name__}Actor"
    EnvironmentActor.__qualname__ = f"{env_class.__qualname__}Actor"

    return EnvironmentActor


def get_env_class(env_name: str | None = None, env_class: str | None = None) -> type[RLEnvironment]:
    """
    Get an environment class by name or class path.

    Args:
        env_name: Name of registered environment (e.g., "wordle")
        env_class: Full class path (e.g., "mymodule.MyEnv")

    Returns:
        The environment class

    Raises:
        ValueError: If neither env_name nor env_class provided, or env not found
    """
    if env_name is not None:
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Environment '{env_name}' not found in registry. " f"Available: {list(ENV_REGISTRY.keys())}")
        return ENV_REGISTRY[env_name]

    if env_class is not None:
        # Import class from full path
        module_path, class_name = env_class.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if not issubclass(cls, RLEnvironment):
            raise ValueError(f"Class {env_class} is not a subclass of RLEnvironment")
        return cls

    raise ValueError("Must provide either env_name or env_class")
