"""
Pure OpenEnv standard interface for RL environments.

This module defines the base RLEnvironment ABC that follows the OpenEnv/Gymnasium-style API.
Concrete implementations should inherit from RLEnvironment and implement reset() and step().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """OpenEnv standard result from reset() and step() calls."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class RLEnvironment(ABC):
    """
    Pure OpenEnv standard - Gymnasium-style APIs.

    Concrete environments should implement:
    - reset(): Initialize episode, return initial observation
    - step(): Execute action, return observation/reward/done

    Optional overrides:
    - state(): Return episode metadata
    - close(): Cleanup resources
    """

    @abstractmethod
    def reset(self) -> StepResult:
        """Initialize episode, return initial observation."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute action, return observation/reward/done."""

    def state(self) -> dict[str, Any]:
        """Return episode metadata (step count, etc.)."""
        return {}

    def close(self) -> None:
        """Cleanup resources. Override in subclasses that need cleanup."""
        return None  # Default no-op; subclasses may override
