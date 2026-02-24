"""SWERL Sandbox environment — per-sample Docker tasks with submit-based evaluation.

Provides execute_bash, str_replace_editor, and submit tools inside a Docker container.

Each task has its own files on disk at ``{task_data_dir}/{task_id}/``:
- ``instruction.md`` — task description (returned as observation on reset)
- ``tests/`` — test files copied into the container (``test.sh`` is the entrypoint)
- ``environment/seeds/`` — seed files copied to ``/workspace/``
- ``image.txt`` — (optional) Docker image tag to use for this task
- ``setup.sh`` — (optional) setup script run after seeding
"""

import contextlib
import os
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
    pass


_BASH_WRAPPER = r"""#!/bin/bash
set -a; source /tmp/.sandbox_env 2>/dev/null; set +a
cd "$(cat /tmp/.sandbox_cwd 2>/dev/null || echo /workspace)" 2>/dev/null
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

_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your solution and run the test suite. Only call when you believe the task is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []},
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


class SWERLSandboxEnv(RLEnvironment):
    """Environment for per-sample coding tasks with Docker-based evaluation.

    Provides three tools:
    * ``execute_bash`` — stateful bash (env vars, cwd persist between calls)
    * ``str_replace_editor`` — file viewer/editor (view/create/str_replace/insert)
    * ``submit`` — runs per-task test script and returns the reward
    """

    config_name = "swerl_sandbox"
    _tool_definitions = (_EXECUTE_BASH_TOOL, _STR_REPLACE_EDITOR_TOOL, _SUBMIT_TOOL)

    def __init__(
        self,
        backend: str = "docker",
        image: str = "python:3.12-slim",
        task_data_dir: str = "",
        test_timeout: int = 120,
        timeout: int = 600,
        **backend_kwargs: Any,
    ):
        backend_kwargs["image"] = image
        self._backend_type = backend
        self._timeout = timeout
        self._penalty = backend_kwargs.pop("penalty", -0.05)
        backend_kwargs.pop("call_name", None)
        self._backend_kwargs = backend_kwargs
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._task_id: str | None = None
        self._task_data_dir = task_data_dir
        self._test_timeout = test_timeout
        self._instruction = ""

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        last_error = None
        for attempt in range(3):
            try:
                return self._do_reset(task_id, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"SWERLSandboxEnv.reset attempt {attempt + 1} failed: {e}. Retrying...")
                if self._backend is not None:
                    with contextlib.suppress(Exception):
                        self._backend.close()
                    self._backend = None
        if last_error is not None:
            raise RuntimeError(f"Reset failed after 3 attempts: {last_error}") from last_error
        else:
            raise RuntimeError("Reset failed without capturing an error.")

    def _do_reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        per_sample_config = kwargs.get("env_config") or {}

        # Image priority: per-sample env_config > image.txt on disk > default from __init__
        env_image = per_sample_config.get("image")
        if env_image:
            self._backend_kwargs["image"] = env_image
        elif self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
            image_file = os.path.join(task_dir, "image.txt")
            if os.path.isfile(image_file):
                with open(image_file) as f:
                    image_tag = f.read().strip()
                if image_tag:
                    self._backend_kwargs["image"] = image_tag

        if self._backend is not None:
            self._backend.close()
        bkwargs = dict(self._backend_kwargs)
        bkwargs.setdefault("timeout", self._timeout)
        self._backend = create_backend(self._backend_type, **bkwargs)
        self._backend.start()
        self._step_count = 0
        self._task_id = task_id

        # Set up workspace
        self._backend.run_command("mkdir -p /workspace")
        self._backend.write_file("/tmp/.sandbox_bash_wrapper.sh", _BASH_WRAPPER)
        self._backend.run_command("chmod +x /tmp/.sandbox_bash_wrapper.sh")
        self._backend.run_command("echo /workspace > /tmp/.sandbox_cwd")
        self._backend.run_command("mkdir -p /output /logs/verifier")

        # Load task data if available
        self._instruction = ""
        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
        if task_dir and os.path.isdir(task_dir):
            self._load_task_data(task_dir)

        observation = self._instruction or "Sandbox ready."
        if not self._instruction and task_id:
            observation = f"[Task: {task_id}] {observation}"

        return (
            StepResult(result=observation, metadata={"task_id": task_id, "backend": self._backend_type}),
            list(self._tool_definitions),
        )

    def _load_task_data(self, task_dir: str) -> None:
        """Load instruction, seeds, and tests from the task directory into the container."""
        assert self._backend is not None

        instruction_file = os.path.join(task_dir, "instruction.md")
        if os.path.isfile(instruction_file):
            with open(instruction_file) as f:
                self._instruction = f.read().strip()

        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            for root, _dirs, files in os.walk(seeds_dir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(src_path, seeds_dir)
                    container_path = f"/workspace/{rel_path}"
                    parent = os.path.dirname(container_path)
                    if parent != "/workspace":
                        self._backend.run_command(f"mkdir -p {parent}")
                    with open(src_path) as f:
                        content = f.read()
                    self._backend.write_file(container_path, content)

        tests_dir = os.path.join(task_dir, "tests")
        if os.path.isdir(tests_dir):
            self._backend.run_command("mkdir -p /tests")
            for root, _dirs, files in os.walk(tests_dir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(src_path, tests_dir)
                    container_path = f"/tests/{rel_path}"
                    parent = os.path.dirname(container_path)
                    if parent != "/tests":
                        self._backend.run_command(f"mkdir -p {parent}")
                    with open(src_path) as f:
                        content = f.read()
                    self._backend.write_file(container_path, content)
            self._backend.run_command("chmod +x /tests/test.sh 2>/dev/null || true")

        setup_file = os.path.join(task_dir, "setup.sh")
        if os.path.isfile(setup_file):
            with open(setup_file) as f:
                setup_content = f.read()
            self._backend.write_file("/tmp/setup.sh", setup_content)
            self._backend.run_command("chmod +x /tmp/setup.sh && bash /tmp/setup.sh")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
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
        elif call.name == "submit":
            return self._run_tests()
        else:
            return StepResult(
                result=f"Error: Unknown tool '{call.name}'. Available: execute_bash, str_replace_editor, submit",
                reward=self._penalty,
            )

    # ------------------------------------------------------------------
    # execute_bash
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # submit (test execution)
    # ------------------------------------------------------------------
    def _run_tests(self) -> StepResult:
        """Run the task's test script and return the reward."""
        assert self._backend is not None

        check = self._backend.run_command("test -f /tests/test.sh && echo EXISTS")
        if check.stdout.strip() != "EXISTS":
            return StepResult(result="No test script found at /tests/test.sh. Reward: 0.0", reward=0.0, done=True)

        result = self._backend.run_command(f"timeout {self._test_timeout} bash /tests/test.sh")

        reward = self._parse_reward(result.exit_code)

        stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        stderr = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr

        observation = (
            f"Test execution complete.\n"
            f"Exit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}\n"
            f"Reward: {reward}"
        )

        return StepResult(result=observation, reward=reward, done=True)

    def _parse_reward(self, exit_code: int) -> float:
        """Parse reward from /logs/verifier/reward.txt, falling back to exit code."""
        assert self._backend is not None

        try:
            reward_result = self._backend.run_command("cat /logs/verifier/reward.txt")
            if reward_result.exit_code == 0 and reward_result.stdout.strip():
                reward = float(reward_result.stdout.strip())
                return max(0.0, min(1.0, reward))
        except (ValueError, TypeError):
            pass

        return 1.0 if exit_code == 0 else 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    def state(self) -> State:
        return State(episode_id=self._task_id, step_count=self._step_count)

    async def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    async def shutdown(self) -> None:
        await self.close()


@dataclass
class SWERLSandboxEnvConfig(BaseEnvConfig):
    """Configuration for SWERLSandboxEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = SWERLSandboxEnv
    backend: str = "docker"
    image: str = "python:3.12-slim"
    mem_limit: str = "4g"
    penalty: float = -0.05
    task_data_dir: str = ""
    test_timeout: int = 120
    timeout: int = 600
