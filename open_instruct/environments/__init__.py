"""
RL Environments for open-instruct.

This module provides:
- RLEnvironment: Abstract base class for environments (Ray actor)
- ResetResult, StepResult, ToolCall: Data classes for env interaction
- EnvironmentPool: Pool manager for environment actors
- SandboxBackend: Abstract backend for code execution (E2B, Docker)
- Concrete environments: SandboxEnv, OpenEnvClient, etc.

Usage:
    from open_instruct.environments import (
        RLEnvironment,
        ResetResult,
        StepResult,
        ToolCall,
        EnvironmentPool,
        ENV_REGISTRY,
        register_env,
    )

    # Create environment pool
    pool = EnvironmentPool(
        env_name="sandbox",
        pool_size=64,
        backend="e2b",
    )
    await pool.initialize()

    # Acquire and use environment
    env = await pool.acquire()
    result = await env.reset.remote(task_id="task_123")
    step = await env.step.remote(ToolCall(name="execute", args={"command": "ls"}))
    pool.release(env)
"""

from .base import (
    ENV_REGISTRY,
    EnvironmentState,
    ResetResult,
    RLEnvironment,
    StepResult,
    ToolCall,
    get_env_class,
    make_env_actor,
    register_env,
)
from .pool import EnvironmentPool

# Import concrete implementations to register them
from . import openenv_client  # noqa: F401
from . import sandbox  # noqa: F401
from . import examples  # noqa: F401

# Import backends
from .backends import (
    DockerBackend,
    E2BBackend,
    ExecutionResult,
    SandboxBackend,
    create_backend,
)

__all__ = [
    # Base classes
    "RLEnvironment",
    "ResetResult",
    "StepResult",
    "ToolCall",
    "EnvironmentState",
    # Registry
    "ENV_REGISTRY",
    "register_env",
    "get_env_class",
    "make_env_actor",
    # Pool
    "EnvironmentPool",
    # Backends
    "SandboxBackend",
    "E2BBackend",
    "DockerBackend",
    "ExecutionResult",
    "create_backend",
]
