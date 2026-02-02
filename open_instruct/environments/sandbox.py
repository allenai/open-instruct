"""General-purpose sandbox environment with pluggable backends."""

import logging
from typing import Any

from .backends import SandboxBackend, create_backend
from .base import RLEnvironment, StepResult, ToolCall, register_env

logger = logging.getLogger(__name__)


@register_env("sandbox")
class SandboxEnv(RLEnvironment):
    """
    Sandbox environment providing execute, run_code, and submit tools.

    Backend can be "e2b" (cloud, no Docker needed) or "docker" (local).
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
        self._backend_type = backend
        self._backend_kwargs = backend_kwargs
        self._task_prompt = task_prompt
        self._input_files = input_files or {}
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._submitted_answer: str | None = None

    async def reset(self, task_id: str | None = None) -> StepResult:
        """Initialize the sandbox for a new episode."""
        if self._backend is not None:
            self._backend.close()

        self._backend = create_backend(self._backend_type, **self._backend_kwargs)
        self._backend.start()

        self._step_count = 0
        self._submitted_answer = None

        for path, content in self._input_files.items():
            self._backend.write_file(path, content)

        observation = f"Sandbox ready. Task: {self._task_prompt}"
        if task_id:
            observation = f"[Task: {task_id}] {observation}"

        return StepResult(
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
        """Execute an action in the sandbox."""
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if tool_call.name == "submit":
            self._submitted_answer = tool_call.args.get("answer", "")
            return StepResult(
                observation=f"Submitted: {self._submitted_answer}",
                reward=0.0,
                done=True,
                info={"answer": self._submitted_answer},
            )

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

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if not output.strip():
            output = "(no output)"

        reward = 0.0 if result.exit_code == 0 else -0.05

        return StepResult(observation=output, reward=reward, done=False, info={"exit_code": result.exit_code})

    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count), "submitted": 1.0 if self._submitted_answer is not None else 0.0}

    async def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
