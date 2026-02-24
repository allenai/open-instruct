"""Generic sandbox environment — provides execute_bash + str_replace_editor tools.

Modified from https://github.com/llm-in-sandbox/llm-in-sandbox
"""

import contextlib
import shlex
from dataclasses import dataclass
from typing import Any, ClassVar

from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .backends import SandboxBackend, create_backend
from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .tools.utils import coerce_args

logger = logger_utils.setup_logger(__name__)


class _EditorError(Exception):
    """Raised by editor sub-commands to signal a user-visible error."""


_BASH_WRAPPER = r"""#!/bin/bash
set -a; source /tmp/.sandbox_env 2>/dev/null; set +a
cd "$(cat /tmp/.sandbox_cwd 2>/dev/null || echo /testbed)" 2>/dev/null
eval "$1"
_exit_code=$?
export -p > /tmp/.sandbox_env
pwd > /tmp/.sandbox_cwd
exit $_exit_code
"""

_EXECUTE_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": (
            "Execute a bash command in the terminal.\n"
            "* Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.\n"
            "* Interactive: Not possible. Use `yes`/`no`, etc. as appropriate.\n"
            "* Output: May be truncated. Use `head`/`tail`/`grep` to filter."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

_STR_REPLACE_EDITOR_TOOL = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "Custom editing tool for viewing, creating, and editing files.\n"
            "* State is persistent across command calls and discussions.\n"
            "* `view` for reading files/directories, `create` for new files,\n"
            "  `str_replace` for editing, `insert` for adding lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The editor command to run.",
                },
                "path": {"type": "string", "description": "Absolute path to file or directory."},
                "file_text": {
                    "type": "string",
                    "description": "Required for `create`. The full content of the new file.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Required for `str_replace`. The exact string to replace (must appear exactly once).",
                },
                "new_str": {"type": "string", "description": "Required for `str_replace`/`insert`. The new string."},
                "insert_line": {
                    "type": "integer",
                    "description": "Required for `insert`. Line number after which to insert `new_str`.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional for `view`. Two-element [start, end] line range.",
                },
            },
            "required": ["command", "path"],
        },
    },
}


def _truncate_output(text: str, num_lines: int = 40) -> str:
    """Truncate long output keeping first and last *num_lines* lines."""
    lines = text.splitlines()
    if len(lines) <= 2 * num_lines:
        return text
    top = "\n".join(lines[:num_lines])
    bottom = "\n".join(lines[-num_lines:])
    divider = "-" * 50
    return f"{top}\n{divider}\n<Observation truncated in middle for saving context>\n{divider}\n{bottom}"


class GenericSandboxEnv(RLEnvironment):
    """Generic sandbox environment with execute_bash and str_replace_editor tools.

    Provides two tools:
    * ``execute_bash`` — stateful bash (env vars, cwd persist between calls)
    * ``str_replace_editor`` — file viewer/editor (view/create/str_replace/insert)
    """

    config_name = "generic_sandbox"
    _tool_definitions = (_EXECUTE_BASH_TOOL, _STR_REPLACE_EDITOR_TOOL)

    def __init__(
        self, backend: str = "docker", write_prompt_file: bool = False, timeout: int = 300, **backend_kwargs: Any
    ):
        self._backend_type = backend
        self._write_prompt_file = write_prompt_file
        self._timeout = timeout
        self._penalty = backend_kwargs.pop("penalty", -0.05)
        backend_kwargs.pop("call_name", None)
        self._backend_kwargs = backend_kwargs
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._task_id: str | None = None

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        last_error = None
        for attempt in range(3):
            try:
                return self._do_reset(task_id, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"GenericSandboxEnv.reset attempt {attempt + 1} failed: {e}. Retrying...")
                if self._backend is not None:
                    with contextlib.suppress(Exception):
                        self._backend.close()
                    self._backend = None
        if last_error is not None:
            raise RuntimeError(f"Reset failed after 3 attempts: {last_error}") from last_error
        else:
            raise RuntimeError("Reset failed without capturing an error.")

    def _do_reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        if self._backend is not None:
            self._backend.close()
        bkwargs = dict(self._backend_kwargs)
        bkwargs.setdefault("timeout", self._timeout)
        self._backend = create_backend(self._backend_type, **bkwargs)
        self._backend.start()
        self._step_count = 0
        self._task_id = task_id

        self._backend.run_command("mkdir -p /testbed/input /testbed/output")
        if self._backend.run_command("which git").exit_code == 0:
            git_result = self._backend.run_command(
                "cd /testbed && git init && git config user.email 'tulu@example.com' && git config user.name 'Tulu'"
            )
            if git_result.exit_code != 0:
                logger.warning(f"git init/config failed (exit {git_result.exit_code}): {git_result.stderr.strip()}")
        self._backend.write_file("/tmp/.sandbox_bash_wrapper.sh", _BASH_WRAPPER)
        self._backend.run_command("chmod +x /tmp/.sandbox_bash_wrapper.sh")
        self._backend.run_command("echo /testbed > /tmp/.sandbox_cwd")

        if self._write_prompt_file and task_id:
            self._backend.write_file("/root/prompt.txt", task_id)

        observation = "Sandbox ready."
        if task_id:
            observation = f"[Task: {task_id}] {observation}"

        return (
            StepResult(result=observation, metadata={"task_id": task_id, "backend": self._backend_type}),
            list(self._tool_definitions),
        )

    async def step(self, call: EnvCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name == "execute_bash":
            args = coerce_args(_EXECUTE_BASH_TOOL["function"]["parameters"], call.args)
            return self._execute_bash(args)
        elif call.name == "str_replace_editor":
            args = coerce_args(_STR_REPLACE_EDITOR_TOOL["function"]["parameters"], call.args)
            return self._execute_editor(args)
        else:
            return StepResult(
                result=f"Error: Unknown tool '{call.name}'. Available: execute_bash, str_replace_editor",
                reward=self._penalty,
            )

    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        if not command:
            return StepResult(result="Error: 'command' parameter is required for execute_bash.", reward=self._penalty)

        result = self._backend.run_command(f"bash /tmp/.sandbox_bash_wrapper.sh {shlex.quote(command)}")

        stdout = _truncate_output(result.stdout) if result.stdout else ""
        stderr = _truncate_output(result.stderr) if result.stderr else ""

        observation = (
            f"Exit code: {result.exit_code}\n"
            f"Execution output of [execute_bash]:\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}"
        )

        reward = 0.0 if result.exit_code == 0 else self._penalty
        return StepResult(result=observation, reward=reward, metadata={"exit_code": result.exit_code})

    def _execute_editor(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        path = args.get("path", "")

        if not command:
            return self._editor_error("'command' parameter is required.")
        if not path:
            return self._editor_error("'path' parameter is required.")

        try:
            if command == "view":
                output = self._editor_view(path, args.get("view_range"))
            elif command == "create":
                output = self._editor_create(path, args.get("file_text"))
            elif command == "str_replace":
                output = self._editor_str_replace(path, args.get("old_str"), args.get("new_str"))
            elif command == "insert":
                output = self._editor_insert(path, args.get("insert_line"), args.get("new_str"))
            else:
                return self._editor_error(f"Unknown command '{command}'. Use view/create/str_replace/insert.")
        except (_EditorError, FileNotFoundError) as exc:
            return self._editor_error(str(exc))

        return StepResult(result=f"Execution output of [str_replace_editor]:\n{output}")

    def _editor_view(self, path: str, view_range: list[int] | None = None) -> str:
        assert self._backend is not None
        check = self._backend.run_command(f"test -d {shlex.quote(path)}")
        is_dir = check.exit_code == 0

        if is_dir:
            result = self._backend.run_command(
                f'find {shlex.quote(path)} -maxdepth 2 -not -path "*/.*" | sort | head -100'
            )
            return _truncate_output(result.stdout)

        if view_range and len(view_range) == 2:
            start, end = view_range
            cmd = f"cat -n {shlex.quote(path)} | sed -n '{start},{end}p'"
        else:
            cmd = f"cat -n {shlex.quote(path)}"

        result = self._backend.run_command(cmd)
        if result.exit_code != 0:
            raise _EditorError(f"Failed to view '{path}': {result.stderr or result.stdout}")
        return _truncate_output(result.stdout)

    def _editor_create(self, path: str, file_text: str | None) -> str:
        assert self._backend is not None
        if file_text is None:
            raise _EditorError("'file_text' parameter is required for create.")

        if self._backend.run_command(f"test -d {shlex.quote(path)}").exit_code == 0:
            raise _EditorError(f"'{path}' is a directory. Cannot create a file with the same name.")
        if self._backend.run_command(f"test -e {shlex.quote(path)}").exit_code == 0:
            raise _EditorError(f"File '{path}' already exists. Use str_replace to edit.")

        parent = "/".join(path.rsplit("/", 1)[:-1]) or "/"
        self._backend.run_command(f"mkdir -p {shlex.quote(parent)}")
        self._backend.write_file(path, file_text)
        return f"File created successfully at: {path}"

    def _editor_str_replace(self, path: str, old_str: str | None, new_str: str | None) -> str:
        assert self._backend is not None
        if old_str is None:
            raise _EditorError("'old_str' parameter is required for str_replace.")
        if new_str is None:
            raise _EditorError("'new_str' parameter is required for str_replace.")

        content = self._backend.read_file(path)
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        count = content.count(old_str)
        if count == 0:
            raise _EditorError(f"old_str not found in {path}. Make sure it matches exactly.")
        if count > 1:
            raise _EditorError(f"old_str found {count} times in {path}. It must appear exactly once.")

        new_content = content.replace(old_str, new_str, 1)
        self._backend.write_file(path, new_content)
        return f"The file {path} has been edited. Review the changes with `view`."

    def _editor_insert(self, path: str, insert_line: int | None, new_str: str | None) -> str:
        assert self._backend is not None
        if insert_line is None:
            raise _EditorError("'insert_line' parameter is required for insert.")
        if new_str is None:
            raise _EditorError("'new_str' parameter is required for insert.")

        content = self._backend.read_file(path)
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        lines = content.splitlines(keepends=True)
        idx = max(0, min(insert_line, len(lines)))
        insert_text = new_str if new_str.endswith("\n") else new_str + "\n"
        insert_lines = insert_text.splitlines(keepends=True)
        new_lines = lines[:idx] + insert_lines + lines[idx:]
        self._backend.write_file(path, "".join(new_lines))
        return f"The file {path} has been edited. Review the changes with `view`."

    def _editor_error(self, message: str) -> StepResult:
        return StepResult(result=f"Execution output of [str_replace_editor]:\nERROR: {message}", reward=self._penalty)

    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    def state(self) -> State:
        return State(episode_id=self._task_id, step_count=self._step_count)

    async def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    async def shutdown(self) -> None:
        """Clean up backend on actor shutdown."""
        await self.close()


@dataclass
class GenericSandboxEnvConfig(BaseEnvConfig):
    """Configuration for GenericSandboxEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = GenericSandboxEnv
    backend: str = "docker"
    image: str = "python:3.12-slim"
    mem_limit: str = "4g"
    penalty: float = -0.05
    write_prompt_file: bool = False
    timeout: int = 300
