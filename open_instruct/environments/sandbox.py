"""
General-purpose sandbox environment.

Provides code and command execution via pluggable backends (E2B or Docker).
Supports file I/O and submission of final answers.
"""

import logging
from typing import Any

from .backends import SandboxBackend, create_backend
from .base import ResetResult, RLEnvironment, StepResult, ToolCall, register_env

logger = logging.getLogger(__name__)


@register_env("sandbox")
class SandboxEnv(RLEnvironment):
    """
    General sandbox environment with pluggable backend.

    Provides three tools:
    - execute: Run bash commands
    - run_code: Execute Python code
    - submit: Submit final answer and end episode

    Backend can be:
    - "e2b": E2B cloud sandbox (default, no Docker needed)
    - "docker": Local Docker via llm-in-sandbox

    Usage:
        # Create actor
        env = SandboxEnv.remote(
            backend="e2b",
            task_prompt="Analyze data and create visualization",
            input_files={"/sandbox/data.csv": "name,value\\nA,10\\nB,20"}
        )

        # Run episode
        result = await env.reset.remote(task_id="task_123")
        step = await env.step.remote(ToolCall(name="run_code", args={"code": "print(1+1)"}))
    """

    response_role = "tool"
    max_steps = 50

    def __init__(
        self,
        backend: str = "e2b",
        task_prompt: str = "",
        input_files: dict[str, str] | None = None,
        **backend_kwargs: Any,
    ):
        """
        Initialize sandbox environment.

        Args:
            backend: Backend type - "e2b" (default) or "docker"
            task_prompt: Task description shown to model
            input_files: Dict of path -> content to write on reset
            **backend_kwargs: Backend-specific arguments (template, timeout, image, etc.)
        """
        self._backend_type = backend
        self._backend_kwargs = backend_kwargs
        self._task_prompt = task_prompt
        self._input_files = input_files or {}
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._submitted_answer: str | None = None

    async def reset(self, task_id: str | None = None) -> ResetResult:
        """
        Initialize the sandbox for a new episode.

        Starts the backend, writes input files, and returns tools.

        Args:
            task_id: Optional task identifier (for logging/tracking)

        Returns:
            ResetResult with initial observation and available tools
        """
        # Close any existing backend
        if self._backend is not None:
            self._backend.close()

        # Create and start new backend
        self._backend = create_backend(self._backend_type, **self._backend_kwargs)
        self._backend.start()

        # Reset state
        self._step_count = 0
        self._submitted_answer = None

        # Write input files
        for path, content in self._input_files.items():
            self._backend.write_file(path, content)

        # Build observation
        observation = f"Sandbox ready. Task: {self._task_prompt}"
        if task_id:
            observation = f"[Task: {task_id}] {observation}"

        return ResetResult(
            observation=observation,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "description": "Execute a bash command in the sandbox",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string", "description": "Bash command to execute"}},
                            "required": ["command"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_code",
                        "description": "Execute Python code in the sandbox",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                            "required": ["code"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "submit",
                        "description": "Submit your final answer and end the episode",
                        "parameters": {
                            "type": "object",
                            "properties": {"answer": {"type": "string", "description": "Your final answer"}},
                            "required": ["answer"],
                        },
                    },
                },
            ],
            info={"task_id": task_id, "backend": self._backend_type},
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """
        Execute an action in the sandbox.

        Args:
            tool_call: Parsed tool call (execute, run_code, or submit)

        Returns:
            StepResult with observation, reward, and done flag
        """
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        # Handle submission
        if tool_call.name == "submit":
            self._submitted_answer = tool_call.args.get("answer", "")
            return StepResult(
                observation=f"Submitted: {self._submitted_answer}",
                reward=0.0,  # Reward determined by verifier
                done=True,
                info={"answer": self._submitted_answer},
            )

        # Execute based on tool type
        if tool_call.name == "run_code":
            code = tool_call.args.get("code", "")
            result = self._backend.run_code(code)
        elif tool_call.name == "execute":
            command = tool_call.args.get("command", "")
            result = self._backend.run_command(command)
        else:
            return StepResult(
                observation=f"Error: Unknown tool '{tool_call.name}'. Available: execute, run_code, submit",
                reward=-0.05,
                done=False,
            )

        # Build observation
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if not output.strip():
            output = "(no output)"

        # Small penalty for errors
        reward = 0.0 if result.exit_code == 0 else -0.05

        return StepResult(observation=output, reward=reward, done=False, info={"exit_code": result.exit_code})

    def get_metrics(self) -> dict[str, float]:
        """Return sandbox-specific metrics."""
        return {"step_count": float(self._step_count), "submitted": 1.0 if self._submitted_answer is not None else 0.0}

    async def close(self) -> None:
        """Cleanup the sandbox backend."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
