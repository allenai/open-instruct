"""
RL Environment support for open-instruct.

This module provides a generic interface for RL environments following the OpenEnv standard,
with adapters for Prime Intellect verifiers envs and AppWorld.
"""

from open_instruct.environments.base import RLEnvironment, StepResult

__all__ = ["RLEnvironment", "StepResult"]
