"""SandboxLM environment — mirrors llm-in-sandbox tools (execute_bash + str_replace_editor)."""

import contextlib
import shlex

from open_instruct import logger_utils

from .backends import SandboxBackend, create_backend
from .base import RLEnvironment, StepResult, ToolCall, register_env

logger = logger_utils.setup_logger(__name__)


class _EditorError(Exception):
    """Raised by editor sub-commands to signal a user-visible error."""


# ---------------------------------------------------------------------------
# Bash wrapper (written once to the sandbox in reset())
# ---------------------------------------------------------------------------
_BASH_WRAPPER = r"""#!/bin/bash
set -a; source /tmp/.sandbox_env 2>/dev/null; set +a
cd "$(cat /tmp/.sandbox_cwd 2>/dev/null || echo /testbed)" 2>/dev/null
eval "$1"
_exit_code=$?
export -p > /tmp/.sandbox_env
pwd > /tmp/.sandbox_cwd
exit $_exit_code
"""

# ---------------------------------------------------------------------------
# Tool schemas (mirror llm_in_sandbox/tools.py)
# ---------------------------------------------------------------------------
_EXECUTE_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": (
            "Execute a bash command in the terminal.\n"
            "* Long running commands: Coverage with `timeout`, e.g., `timeout 10 <command>`.\n"
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@register_env("sandbox_lm")
class SandboxLMEnv(RLEnvironment):
    """Sandbox environment compatible with the *llm-in-sandbox* tool interface.

    Provides two tools:
    * ``execute_bash`` — stateful bash (env vars, cwd persist between calls)
    * ``str_replace_editor`` — file viewer/editor (view/create/str_replace/insert)
    """

    response_role = "tool"
    max_steps = 100

    _tool_definitions = [_EXECUTE_BASH_TOOL, _STR_REPLACE_EDITOR_TOOL]

    def __init__(self, backend: str = "docker", write_prompt_file: bool = False, timeout: int = 300, **backend_kwargs):
        self._backend_type = backend
        self._write_prompt_file = write_prompt_file
        self._timeout = timeout
        self._backend_kwargs = backend_kwargs
        self._backend: SandboxBackend | None = None
        self._step_count = 0

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        last_error = None
        for attempt in range(3):
            try:
                return self._do_reset(task_id, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"SandboxLMEnv.reset attempt {attempt + 1} failed: {e}. Retrying...")
                # Backend may be stale — force close before retry
                if self._backend is not None:
                    with contextlib.suppress(Exception):
                        self._backend.close()
                    self._backend = None
        raise last_error  # type: ignore[misc]

    def _do_reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        if self._backend is not None:
            self._backend.reset()
        else:
            bkwargs = dict(self._backend_kwargs)
            if self._backend_type == "e2b":
                bkwargs.setdefault("timeout", self._timeout)
            self._backend = create_backend(self._backend_type, **bkwargs)
            self._backend.start()
        self._step_count = 0

        # Setup env
        self._backend.run_command("mkdir -p /testbed/input /testbed/output")
        self._backend.run_command("cd /testbed && git init 2>/dev/null || true")

        # Write stateful bash wrapper
        self._backend.write_file("/tmp/.sandbox_bash_wrapper.sh", _BASH_WRAPPER)
        self._backend.run_command("chmod +x /tmp/.sandbox_bash_wrapper.sh")

        # Init cwd state
        self._backend.run_command("echo /testbed > /tmp/.sandbox_cwd")

        # Optionally write the prompt/task to a file in the sandbox
        if self._write_prompt_file and task_id:
            self._backend.write_file("/root/prompt.txt", task_id)

        observation = "Sandbox ready."
        if task_id:
            observation = f"[Task: {task_id}] {observation}"

        return StepResult(
            observation=observation,
            tools=self._tool_definitions,
            info={"task_id": task_id, "backend": self._backend_type},
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    async def step(self, tool_call: ToolCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if tool_call.name == "execute_bash":
            return self._execute_bash(tool_call.args)
        elif tool_call.name == "str_replace_editor":
            return self._execute_editor(tool_call.args)
        else:
            return StepResult(
                observation=(f"Error: Unknown tool '{tool_call.name}'. Available: execute_bash, str_replace_editor"),
                reward=-0.05,
                done=False,
            )

    # ------------------------------------------------------------------
    # execute_bash
    # ------------------------------------------------------------------
    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        if not command:
            return StepResult(
                observation="Error: 'command' parameter is required for execute_bash.", reward=-0.05, done=False
            )

        result = self._backend.run_command(f"bash /tmp/.sandbox_bash_wrapper.sh {shlex.quote(command)}")

        stdout = _truncate_output(result.stdout) if result.stdout else ""
        stderr = _truncate_output(result.stderr) if result.stderr else ""

        observation = (
            f"Exit code: {result.exit_code}\n"
            f"Execution output of [execute_bash]:\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}"
        )

        reward = 0.0 if result.exit_code == 0 else -0.05
        return StepResult(observation=observation, reward=reward, done=False, info={"exit_code": result.exit_code})

    # ------------------------------------------------------------------
    # str_replace_editor
    # ------------------------------------------------------------------
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
        except _EditorError as exc:
            return self._editor_error(str(exc))

        observation = f"Execution output of [str_replace_editor]:\n{output}"
        return StepResult(observation=observation, reward=0.0, done=False)

    # -- view -------------------------------------------------------
    def _editor_view(self, path: str, view_range: list[int] | None = None) -> str:
        assert self._backend is not None
        # Check if path is a directory
        check = self._backend.run_command(f"test -d {shlex.quote(path)} && echo DIR || echo FILE")
        is_dir = check.stdout.strip().startswith("DIR")

        if is_dir:
            result = self._backend.run_command(
                f'find {shlex.quote(path)} -maxdepth 2 -not -path "*/.*" | sort | head -100'
            )
            return _truncate_output(result.stdout)

        # File view
        if view_range and len(view_range) == 2:
            start, end = view_range
            cmd = f"sed -n '{start},{end}p' {shlex.quote(path)} | cat -n"
        else:
            cmd = f"cat -n {shlex.quote(path)}"

        result = self._backend.run_command(cmd)
        if result.exit_code != 0:
            raise _EditorError(f"Failed to view '{path}': {result.stderr or result.stdout}")
        return _truncate_output(result.stdout)

    # -- create -----------------------------------------------------
    def _editor_create(self, path: str, file_text: str | None) -> str:
        assert self._backend is not None
        if file_text is None:
            raise _EditorError("'file_text' parameter is required for create.")

        # Check file does not already exist
        check = self._backend.run_command(f"test -e {shlex.quote(path)} && echo EXISTS")
        if check.stdout.strip() == "EXISTS":
            raise _EditorError(f"File '{path}' already exists. Use str_replace to edit.")

        # Ensure parent directory exists
        parent = "/".join(path.rsplit("/", 1)[:-1]) or "/"
        self._backend.run_command(f"mkdir -p {shlex.quote(parent)}")

        self._backend.write_file(path, file_text)
        return f"File created successfully at: {path}"

    # -- str_replace -------------------------------------------------
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

    # -- insert ------------------------------------------------------
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

        # Clamp line number
        idx = max(0, min(insert_line, len(lines)))
        insert_lines = new_str.splitlines(keepends=True)
        # Ensure trailing newline on inserted text
        if insert_lines and not insert_lines[-1].endswith("\n"):
            insert_lines[-1] += "\n"

        new_lines = lines[:idx] + insert_lines + lines[idx:]
        self._backend.write_file(path, "".join(new_lines))
        return f"The file {path} has been edited. Review the changes with `view`."

    # -- helpers -----------------------------------------------------
    @staticmethod
    def _editor_error(message: str) -> StepResult:
        return StepResult(
            observation=f"Execution output of [str_replace_editor]:\nERROR: {message}", reward=-0.05, done=False
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    async def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    async def shutdown(self) -> None:
        """Clean up backend on actor shutdown."""
        await self.close()
