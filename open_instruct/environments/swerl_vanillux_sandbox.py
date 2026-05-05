"""SWERL sandbox environment with the Vanillux/SWE-agent tool setup.

This keeps SWERL's per-task Docker setup and verifier flow, but exposes the
upstream SWE-agent default tool surface used by ``VanilluxAgent``:

* stateful ``bash``
* ``str_replace_editor`` with ``view/create/str_replace/insert/undo_edit``
* first-class ``submit``
"""

import ast
import asyncio
import contextlib
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from huggingface_hub import snapshot_download
from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .backends import SandboxBackend, SandboxOOMError, create_backend
from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .tools.utils import coerce_args

logger = logger_utils.setup_logger(__name__)

SWE_AGENT_ENV_FILE = "/root/.swe-agent-env"
SWE_AGENT_STATE_FILE = "/root/state.json"
SUBMIT_REVIEW_MESSAGE = """Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.

1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
If the reproduction script is failing, please revisit your changes and make sure they are correct.
If you have already removed your reproduction script, please ignore this step.
2. Remove your reproduction script (if you haven't done so already).
3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
You can do this with `git checkout -- /path/to/test/file.py`. Use below to find the files you need to revert.
4. Run the submit command again to confirm.

Here is a list of all of your changes:


{{diff}}
"""
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
LAST_STEP_WARNING = "Warning: you only have one more tool call remaining. You must call the `submit` tool next."
TRUNCATED_MESSAGE = (
    " To save on context only part of this file has been shown to you. You should retry this tool after you have "
    "searched inside the file with `grep -n` in order to find the line numbers of what you are looking for. "
)
MAX_RESPONSE_LEN = 16_000
TIMING_LOGS = os.getenv("SWERL_SANDBOX_TIMING_LOGS", "").strip().lower() not in {"", "0", "false", "no", "off"}
TIMING_LOG_THRESHOLD_S = float(os.getenv("SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S", "1.0"))
VANILLUX_CALL_LIMIT = int(os.getenv("VANILLUX_CALL_LIMIT", "100"))

_BASH_WRAPPER_PATH = "/tmp/.swerl_vanillux_bash_wrapper.sh"
_BASH_CWD_PATH = "/tmp/.swerl_vanillux_cwd"
_BASH_ENV_PATH = "/tmp/.swerl_vanillux_env"
_SWE_AGENT_TOOL_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "GIT_PAGER": "cat",
}
_SWE_AGENT_TOOL_ENV_EXPORTS = "\n".join(
    f"export {name}={shlex.quote(value)}" for name, value in _SWE_AGENT_TOOL_ENV.items()
)

_BASH_WRAPPER = f"""#!/bin/bash
set -a
source {shlex.quote(_BASH_ENV_PATH)} 2>/dev/null || true
set +a
{_SWE_AGENT_TOOL_ENV_EXPORTS}
_cwd="$(cat {shlex.quote(_BASH_CWD_PATH)} 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > {shlex.quote(_BASH_ENV_PATH)}
pwd > {shlex.quote(_BASH_CWD_PATH)}
python3 - <<'PY' 2>/dev/null || true
import json
import os
from pathlib import Path

Path({SWE_AGENT_STATE_FILE!r}).write_text(json.dumps({{"working_dir": os.getcwd()}}))
PY
exit $_exit_code
"""

_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a stateful shell. Environment variables and the current working directory "
            "persist across bash calls."
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
            "Custom editing tool for viewing, creating and editing files\n"
            "* State is persistent across command calls and discussions with the user\n"
            "* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, "
            "`view` lists non-hidden files and directories up to 2 levels deep\n"
            "* The `create` command cannot be used if the specified `path` already exists as a file\n"
            "* If a `command` generates a long output, it will be truncated and marked with ` `\n"
            "* The `undo_edit` command will revert the last edit made to the file at `path`\n\n"
            "Notes for using the `str_replace` command:\n"
            "* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. "
            "Be mindful of whitespaces!\n"
            "* If the `old_str` parameter is not unique in the file, the replacement will not be performed. "
            "Make sure to include enough context in `old_str` to make it unique\n"
            "* The `new_str` parameter should contain the edited lines that should replace the `old_str`"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Required parameter of `create` command, with the content of the file to be created.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
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
        "description": "submits the current file",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _truncate(text: str, limit: int = MAX_RESPONSE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATED_MESSAGE


def _as_text(content: str | bytes) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def _make_cat_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
    content = _truncate(file_content).expandtabs()
    numbered = "\n".join(f"{line_no:6}\t{line}" for line_no, line in enumerate(content.split("\n"), init_line))
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n{numbered}\n"


class _EditorError(Exception):
    """Raised by editor subcommands to produce a user-visible tool error."""


class SWERLVanilluxSandboxEnv(RLEnvironment):
    """SWERL sandbox with the Vanillux/SWE-agent default tool surface."""

    config_name = "swerl_vanillux_sandbox"
    _tool_definitions = (_BASH_TOOL, _STR_REPLACE_EDITOR_TOOL, _SUBMIT_TOOL)
    _RESET_RETRY_BASE_DELAY_S = 1.0
    _RESET_RETRY_MAX_DELAY_S = 16.0
    _RESET_RETRY_JITTER_S = 2.0

    def __init__(
        self,
        backend: str = "docker",
        image: str = "python:3.12-slim",
        task_data_dir: str = "",
        task_data_hf_repo: str = "",
        test_timeout: int = 120,
        timeout: int = 600,
        last_step_warning: bool = False,
        **backend_kwargs: Any,
    ):
        backend_kwargs["image"] = image
        self._backend_type = backend
        self._timeout = timeout
        backend_kwargs.pop("penalty", None)
        backend_kwargs.pop("call_name", None)
        self._backend_kwargs = backend_kwargs
        self._backend: SandboxBackend | None = None
        self._step_count = 0
        self._task_id: str | None = None
        self._task_data_dir = task_data_dir
        self._task_data_hf_repo = task_data_hf_repo
        self._test_timeout = test_timeout
        self._last_step_warning = last_step_warning
        self._max_steps: int | None = None
        self._instruction = ""
        self._tests_dir: str | None = None
        self._registry_cache: dict[str, Any] | None = None

    @staticmethod
    def resolve_task_data_dir(task_data_hf_repo: str) -> str:
        """Download and extract task data once per machine."""
        repo_dir = snapshot_download(task_data_hf_repo, repo_type="dataset")
        tarball = os.path.join(repo_dir, "task-data.tar.gz")
        if os.path.isfile(tarball):
            extract_dir = tarball + ".extracted"
            complete_file = os.path.join(extract_dir, ".extraction_complete")
            lock_dir = extract_dir + ".lock"
            while not os.path.isfile(complete_file):
                try:
                    os.mkdir(lock_dir)
                except FileExistsError:
                    time.sleep(1)
                    continue
                try:
                    if os.path.isfile(complete_file):
                        break
                    logger.info(f"Extracting {tarball} to {extract_dir}...")
                    if os.path.isdir(extract_dir):
                        shutil.rmtree(extract_dir)
                    os.makedirs(extract_dir, exist_ok=True)
                    subprocess.run(["tar", "-xzf", tarball, "-C", extract_dir], check=True)
                    with open(complete_file, "w", encoding="utf-8") as f:
                        f.write("ok\n")
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.rmdir(lock_dir)
            return extract_dir
        return repo_dir

    async def setup(self) -> None:
        """Download task data from HuggingFace if configured."""
        if self._task_data_hf_repo and (not self._task_data_dir or not os.path.isdir(self._task_data_dir)):
            if self._task_data_dir:
                logger.info(
                    f"Task data directory {self._task_data_dir} is not available on this actor; "
                    f"downloading from {self._task_data_hf_repo}..."
                )
            else:
                logger.info(f"Downloading task data from {self._task_data_hf_repo}...")
            self._task_data_dir = self.resolve_task_data_dir(self._task_data_hf_repo)
            logger.info(f"Task data at {self._task_data_dir}")

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    def _read_container_json(self, path: str, default: Any) -> Any:
        assert self._backend is not None
        try:
            content = _as_text(self._backend.read_file(path))
            return json.loads(content) if content.strip() else default
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _write_container_json(self, path: str, value: Any) -> None:
        assert self._backend is not None
        self._backend.run_command(f"mkdir -p {shlex.quote(os.path.dirname(path) or '/')}")
        self._backend.write_file(path, json.dumps(value))

    def _read_registry(self) -> dict[str, Any]:
        if self._registry_cache is not None:
            return self._registry_cache
        registry = self._read_container_json(SWE_AGENT_ENV_FILE, {})
        self._registry_cache = registry if isinstance(registry, dict) else {}
        return self._registry_cache

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self._registry_cache = registry
        self._write_container_json(SWE_AGENT_ENV_FILE, registry)

    def _registry_get(self, key: str, default: Any = None) -> Any:
        return self._read_registry().get(key, default)

    def _registry_set(self, key: str, value: Any) -> None:
        registry = self._read_registry()
        registry[key] = value
        self._write_registry(registry)

    def _get_file_history(self) -> dict[str, list[str]]:
        raw_history = self._registry_get("file_history", "{}")
        try:
            history = json.loads(raw_history) if isinstance(raw_history, str) else raw_history
        except json.JSONDecodeError:
            history = {}
        return history if isinstance(history, dict) else {}

    def _set_file_history(self, history: dict[str, list[str]]) -> None:
        self._registry_set("file_history", json.dumps(history))

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        last_error = None
        max_attempts = 5 if self._backend_type == "docker" else 3
        for attempt in range(max_attempts):
            try:
                return self._do_reset(task_id, **kwargs)
            except Exception as e:
                last_error = e
                if self._backend is not None:
                    with contextlib.suppress(Exception):
                        self._backend.close()
                    self._backend = None
                if attempt + 1 == max_attempts:
                    logger.warning(
                        f"SWERLVanilluxSandboxEnv.reset attempt {attempt + 1} failed: {e}. No attempts remain."
                    )
                    break
                delay = self._reset_retry_delay(attempt + 1)
                logger.warning(
                    "SWERLVanilluxSandboxEnv.reset attempt %s failed: %s. Retrying in %.2fs...", attempt + 1, e, delay
                )
                await asyncio.sleep(delay)
        if last_error is not None:
            raise RuntimeError(f"Reset failed after {max_attempts} attempts: {last_error}") from last_error
        raise RuntimeError("Reset failed without capturing an error.")

    @classmethod
    def _reset_retry_delay(cls, attempt: int) -> float:
        backoff = min(cls._RESET_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)), cls._RESET_RETRY_MAX_DELAY_S)
        return backoff + random.uniform(0.0, cls._RESET_RETRY_JITTER_S)

    def _do_reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        reset_start_time = time.perf_counter()
        phase_start_time = reset_start_time
        timings: dict[str, float] = {}

        def record_phase(name: str) -> None:
            nonlocal phase_start_time
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + (now - phase_start_time)
            phase_start_time = now

        resolved_image = kwargs.get("image")
        if not resolved_image and self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
            image_file = os.path.join(task_dir, "image.txt")
            if os.path.isfile(image_file):
                with open(image_file, encoding="utf-8") as f:
                    image_tag = f.read().strip()
                if image_tag:
                    resolved_image = image_tag
        if not resolved_image:
            instance_details = {
                "task_id": task_id,
                "backend_type": self._backend_type,
                "env_name": kwargs.get("env_name"),
                "instance_name": kwargs.get("instance_name"),
                "instance_id": kwargs.get("instance_id"),
                "episode_id": kwargs.get("episode_id"),
            }
            logger.error(
                "Missing explicit image for SWERLVanilluxSandboxEnv reset: %s",
                {k: v for k, v in instance_details.items() if v is not None},
            )
            raise ValueError(
                "SWERLVanilluxSandboxEnv requires an explicit image per task. "
                "Set env_config.image or provide image.txt in task data."
            )
        self._backend_kwargs["image"] = resolved_image
        if self._backend_type == "docker" and kwargs.get("docker_host"):
            self._backend_kwargs["docker_host"] = kwargs["docker_host"]
        record_phase("resolve_image")

        if (
            self._backend is not None
            and self._backend_type == "docker"
            and getattr(self._backend, "_docker_host", None) != self._backend_kwargs.get("docker_host")
        ):
            self._backend.close()
            record_phase("close")
            self._backend = None

        if self._backend is not None:
            self._backend.close()
            record_phase("close")
            self._backend._image = self._backend_kwargs.get("image", self._backend._image)
            self._backend.start()
            record_phase("start")
        else:
            bkwargs = dict(self._backend_kwargs)
            bkwargs.setdefault("timeout", self._timeout)
            self._backend = create_backend(self._backend_type, **bkwargs)
            record_phase("create_backend")
            self._backend.start()
            record_phase("start")

        self._step_count = 0
        self._task_id = task_id
        self._max_steps = kwargs.get("max_steps", VANILLUX_CALL_LIMIT)
        self._backend.run_command("mkdir -p /workspace /output /logs/verifier")
        record_phase("mkdir")

        self._instruction = ""
        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
        if task_dir and os.path.isdir(task_dir):
            self._load_task_data(task_dir)
        record_phase("load_task_data")

        self._prepare_vanillux_runtime()
        record_phase("prepare_vanillux")

        reset_total_s = time.perf_counter() - reset_start_time
        if TIMING_LOGS and reset_total_s >= TIMING_LOG_THRESHOLD_S:
            logger.info(
                "SWERLVanilluxSandboxEnv.reset timing task_id=%s image=%s total=%.3fs phases=%s",
                task_id,
                resolved_image,
                reset_total_s,
                {key: round(value, 3) for key, value in timings.items()},
            )

        observation = self._instruction or "Sandbox ready."
        if not self._instruction and task_id:
            observation = f"[Task: {task_id}] {observation}"

        return (
            StepResult(result=observation, metadata={"task_id": task_id, "backend": self._backend_type}),
            list(self._tool_definitions),
        )

    def _load_task_data(self, task_dir: str) -> None:
        """Load instruction and seed files into the container. Tests are deferred to submit."""
        assert self._backend is not None

        instruction_file = os.path.join(task_dir, "instruction.md")
        if os.path.isfile(instruction_file):
            with open(instruction_file, encoding="utf-8") as f:
                self._instruction = f.read().strip()

        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            self._upload_directory(seeds_dir, "/workspace")

        self._tests_dir = os.path.join(task_dir, "tests") if os.path.isdir(os.path.join(task_dir, "tests")) else None

    def _upload_directory(self, host_dir: str, container_dir: str) -> None:
        assert self._backend is not None
        prefix = container_dir.lstrip("/")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            dir_info = tarfile.TarInfo(name=prefix)
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            tar.addfile(dir_info)

            for root, dirs, files in os.walk(host_dir):
                rel_root = os.path.relpath(root, host_dir)
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root != "." else d
                    info = tarfile.TarInfo(name=f"{prefix}/{rel_dir}")
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    tar.addfile(info)
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_file = os.path.join(rel_root, fname) if rel_root != "." else fname
                    tar.add(src_path, arcname=f"{prefix}/{rel_file}")

        self._backend.put_archive("/", tar_stream.getvalue())

    def _prepare_vanillux_runtime(self) -> None:
        assert self._backend is not None
        self._backend.run_command(
            "mkdir -p /workspace /root && "
            "cd /workspace && "
            '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
            f"printf '%s\\n' /app > {shlex.quote(_BASH_CWD_PATH)} && "
            f": > {shlex.quote(_BASH_ENV_PATH)}"
        )
        self._backend.write_file(_BASH_WRAPPER_PATH, _BASH_WRAPPER)
        self._backend.run_command(f"chmod +x {shlex.quote(_BASH_WRAPPER_PATH)}")
        self._write_registry(
            {
                "ROOT": "/app",
                "PROBLEM_STATEMENT": self._instruction,
                "SUBMIT_REVIEW_MESSAGES": [SUBMIT_REVIEW_MESSAGE],
                "SUBMIT_STAGE": 0,
                "USE_FILEMAP": "true",
                "file_history": "{}",
                **_SWE_AGENT_TOOL_ENV,
            }
        )
        self._write_container_json(SWE_AGENT_STATE_FILE, {"working_dir": "/app"})

    async def step(self, call: EnvCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name in ("bash", "execute_bash"):
            args = coerce_args(_BASH_TOOL["function"]["parameters"], call.args)
            try:
                return self._with_last_step_warning(self._execute_bash(args))
            except SandboxOOMError as e:
                logger.warning(f"[{self._task_id}] sandbox OOM: {e}")
                return StepResult(
                    result=("Sandbox container was killed by the OOM reaper. Ending episode with reward 0."),
                    reward=0.0,
                    done=True,
                    metadata={"oom_killed": True, "task_id": self._task_id},
                )
        if call.name == "str_replace_editor":
            args = coerce_args(_STR_REPLACE_EDITOR_TOOL["function"]["parameters"], call.args)
            return self._with_last_step_warning(self._execute_editor(args))
        if call.name == "submit":
            return self._submit()
        return self._with_last_step_warning(
            StepResult(result=f"Error: Unknown tool '{call.name}'. Available: bash, str_replace_editor, submit")
        )

    def _with_last_step_warning(self, result: StepResult) -> StepResult:
        if (
            self._last_step_warning
            and not result.done
            and self._max_steps is not None
            and self._step_count == self._max_steps - 1
        ):
            result.result = f"{result.result}\n\n{LAST_STEP_WARNING}"
        return result

    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        if not command:
            return StepResult(result="Error: 'command' parameter is required.")

        result = self._backend.run_command(f"bash {shlex.quote(_BASH_WRAPPER_PATH)} {shlex.quote(command)}")
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr

        if SUBMIT_MARKER in output:
            return self._run_tests()

        output = _truncate(output) if output else "(no output)"
        if result.exit_code != 0:
            output = f"{output}\nExit code: {result.exit_code}"
        return StepResult(result=output, metadata={"exit_code": result.exit_code})

    def _execute_editor(self, args: dict) -> StepResult:
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
            elif command == "undo_edit":
                output = self._editor_undo(path)
            else:
                return self._editor_error(
                    f'Unrecognized command {command}. The allowed commands for the str_replace_editor tool are: "view", "create", "str_replace", "insert", "undo_edit"'
                )
        except (_EditorError, FileNotFoundError) as exc:
            return self._editor_error(str(exc))

        return StepResult(result=output or "(no output)")

    def _path_status(self, path: str) -> str:
        assert self._backend is not None
        quoted = shlex.quote(path)
        result = self._backend.run_command(
            f"if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; else echo missing; fi"
        )
        return result.stdout.strip() or "missing"

    def _validate_path(self, command: str, path: str) -> str:
        if not path.startswith("/"):
            suggested_path = f"/app/{path}"
            raise _EditorError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
        status = self._path_status(path)
        if status == "missing" and command != "create":
            raise _EditorError(f"The path {path} does not exist. Please provide a valid path.")
        if status != "missing" and command == "create":
            raise _EditorError(f"File already exists at: {path}. Cannot overwrite files using command `create`.")
        if status == "dir" and command != "view":
            raise _EditorError(
                f"The path {path} is a directory and only the `view` command can be used on directories"
            )
        return status

    def _editor_view(self, path: str, view_range: list[int] | None = None) -> str:
        assert self._backend is not None
        status = self._validate_path("view", path)
        if status == "dir":
            if view_range:
                raise _EditorError("The `view_range` parameter is not allowed when `path` points to a directory.")
            result = self._backend.run_command(f"find {shlex.quote(path)} -maxdepth 2 -not -path '*/\\.*'")
            if result.stderr:
                raise _EditorError(result.stderr)
            return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{result.stdout}\n"

        content = _as_text(self._backend.read_file(path))
        if view_range is not None:
            lines = content.split("\n")
            start, end = self._parse_view_range(view_range, len(lines))
            return _make_cat_output("\n".join(lines[start - 1 : end]), path, init_line=start)

        if (
            path.endswith(".py")
            and len(content) > MAX_RESPONSE_LEN
            and str(self._registry_get("USE_FILEMAP", "false")).lower() == "true"
        ):
            filemap = self._python_filemap(content)
            if filemap:
                filemap = _truncate(filemap.expandtabs())
                return (
                    " This file is too large to display entirely. Showing abbreviated version. "
                    "Please use `str_replace_editor view` with the `view_range` parameter to show selected lines next. "
                    f"\n{filemap}\n"
                    " The above file has been abbreviated. Please use `str_replace editor view` with `view_range` to look at relevant files in detail. "
                )

        return _make_cat_output(content, path)

    def _parse_view_range(self, view_range: list[int], line_count: int) -> tuple[int, int]:
        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            raise _EditorError("Invalid `view_range`. It should be a list of two integers.")
        start, end = view_range
        if start < 1 or start > line_count:
            raise _EditorError(
                f"Invalid `view_range`: {view_range}. Its first element `{start}` should be within the range of lines of the file: {[1, line_count]}"
            )
        if end > line_count:
            raise _EditorError(
                f"Invalid `view_range`: {view_range}. Its second element `{end}` should be smaller than the number of lines in the file: `{line_count}`"
            )
        if end != -1 and end < start:
            raise _EditorError(
                f"Invalid `view_range`: {view_range}. Its second element `{end}` should be larger or equal than its first `{start}`"
            )
        if end == -1:
            end = line_count
        return start, end

    def _python_filemap(self, content: str) -> str:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return _make_cat_output(content, "the file")

        lines = content.splitlines()
        elided: set[int] = set()
        messages: dict[int, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.body or not hasattr(node, "end_lineno"):
                continue
            body_start = node.body[0].lineno
            body_end = int(node.end_lineno or body_start)
            if body_end - body_start < 5:
                continue
            for line_no in range(body_start, body_end + 1):
                elided.add(line_no)
            messages[body_start] = f"... eliding lines {body_start}-{body_end} ..."

        output: list[str] = []
        for line_no, line in enumerate(lines, start=1):
            if line_no in messages:
                output.append(f"{line_no:6d} {messages[line_no]}")
            if line_no not in elided:
                output.append(f"{line_no:6d} {line}")
        return "\n".join(output)

    def _use_linter(self) -> bool:
        return str(self._registry_get("USE_LINTER", "false")).lower() == "true"

    def _flake8(self, path: str) -> str:
        assert self._backend is not None
        if not path.endswith(".py"):
            return ""
        command = self._registry_get(
            "LINT_COMMAND", "flake8 --isolated --select=F821,F822,F831,E111,E112,E113,E999,E902 {file_path}"
        )
        result = self._backend.run_command(str(command).format(file_path=shlex.quote(path)))
        return result.stdout

    def _lint_epilogue(self, path: str, pre_edit_lint: str) -> str:
        if not self._use_linter():
            return ""
        post_edit_lint = self._flake8(path)
        if not post_edit_lint or post_edit_lint == pre_edit_lint:
            return ""
        return (
            "\n\n Your edits have been applied, but the linter has found syntax errors. \n\n"
            f"{post_edit_lint}\n"
            "Please review the changes and make sure they are correct.\n"
            "Edit the file again if necessary.\n"
        )

    def _editor_create(self, path: str, file_text: str | None) -> str:
        assert self._backend is not None
        self._validate_path("create", path)
        if file_text is None:
            raise _EditorError("'file_text' parameter is required for create.")

        parent = path.rsplit("/", 1)[0] or "/"
        if self._path_status(parent) != "dir":
            raise _EditorError(f"The parent directory {parent} does not exist. Please create it first.")
        self._backend.write_file(path, file_text)
        self._push_undo(path, file_text)
        return f"File created successfully at: {path}"

    def _editor_str_replace(self, path: str, old_str: str | None, new_str: str | None) -> str:
        assert self._backend is not None
        self._validate_path("str_replace", path)
        if old_str is None:
            raise _EditorError("'old_str' parameter is required for str_replace.")

        content = _as_text(self._backend.read_file(path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""
        count = content.count(old_str)
        if count == 0:
            raise _EditorError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.")
        if count > 1:
            lines = [idx + 1 for idx, line in enumerate(content.split("\n")) if old_str in line]
            raise _EditorError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )
        if new_str == old_str:
            raise _EditorError(
                f"No replacement was performed, old_str `{old_str}` is the same as new_str `{new_str}`."
            )

        pre_edit_lint = self._flake8(path) if self._use_linter() else ""
        new_content = content.replace(old_str, new_str)
        self._push_undo(path, content)
        self._backend.write_file(path, new_content)

        epilogue = self._lint_epilogue(path, pre_edit_lint)
        replacement_line = content.split(old_str)[0].count("\n")
        start_line = max(1, replacement_line - 4)
        end_line = min(replacement_line + 4 + new_str.count("\n"), len(new_content.splitlines()))
        snippet = "\n".join(new_content.split("\n")[start_line - 1 : end_line])
        return (
            f"The file {path} has been edited. "
            f"{_make_cat_output(snippet, f'a snippet of {path}', start_line)}"
            "Review the changes and make sure they are as expected. Edit the file again if necessary."
            f"{epilogue}"
        )

    def _editor_insert(self, path: str, insert_line: int | None, new_str: str | None) -> str:
        assert self._backend is not None
        self._validate_path("insert", path)
        if insert_line is None:
            raise _EditorError("'insert_line' parameter is required for insert.")
        if new_str is None:
            raise _EditorError("'new_str' parameter is required for insert.")

        content = _as_text(self._backend.read_file(path)).expandtabs()
        new_str = new_str.expandtabs()
        lines = content.split("\n")
        n_lines_file = len(lines)
        if insert_line < 0 or insert_line > n_lines_file:
            raise _EditorError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_content_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
        snippet_lines = (
            lines[max(0, insert_line - 4) : insert_line] + new_str_lines + lines[insert_line : insert_line + 4]
        )
        new_content = "\n".join(new_content_lines)
        self._push_undo(path, content)
        self._backend.write_file(path, new_content)
        snippet = "\n".join(snippet_lines)
        return (
            f"The file {path} has been edited. "
            f"{_make_cat_output(snippet, 'a snippet of the edited file', max(1, insert_line - 4 + 1))}"
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        )

    def _editor_undo(self, path: str) -> str:
        assert self._backend is not None
        self._validate_path("undo_edit", path)
        history = self._get_file_history()
        path_history = history.get(path, [])
        if not path_history:
            raise _EditorError(f"No edit history for {path}.")

        previous = path_history.pop()
        history[path] = path_history
        self._set_file_history(history)
        self._backend.write_file(path, previous)
        return f"Last edit to {path} undone successfully. {_make_cat_output(previous, path)}"

    def _push_undo(self, path: str, previous_content: str) -> None:
        history = self._get_file_history()
        history.setdefault(path, []).append(previous_content)
        self._set_file_history(history)

    def _editor_error(self, message: str) -> StepResult:
        return StepResult(result=message)

    def _submit(self) -> StepResult:
        """Match SWE-agent's review-on-submit flow before running hidden tests."""
        assert self._backend is not None

        review_messages = self._registry_get("SUBMIT_REVIEW_MESSAGES", [SUBMIT_REVIEW_MESSAGE])
        if not isinstance(review_messages, list):
            review_messages = [SUBMIT_REVIEW_MESSAGE]
        current_stage = int(self._registry_get("SUBMIT_STAGE", 0) or 0)

        if current_stage == len(review_messages):
            return self._run_tests()

        patch = self._collect_git_patch()
        message = str(review_messages[current_stage])
        message = message.replace("{{diff}}", patch)
        message = message.replace("{{problem_statement}}", str(self._registry_get("PROBLEM_STATEMENT", "")))
        self._registry_set("SUBMIT_STAGE", current_stage + 1)
        return StepResult(result=message)

    def _collect_git_patch(self) -> str:
        assert self._backend is not None
        repo_root = str(self._registry_get("ROOT", "/app") or "/app")
        result = self._backend.run_command(
            f"cd {shlex.quote(repo_root)} && git add -A >/dev/null 2>/dev/null && git diff --cached || true"
        )
        return result.stdout

    def _run_tests(self) -> StepResult:
        """Upload tests (deferred from reset to prevent peeking), then run."""
        assert self._backend is not None

        if self._tests_dir is None:
            raise RuntimeError(f"No test data directory for task {self._task_id}")

        check_stdout = ""
        for attempt in range(2):
            try:
                self._upload_directory(self._tests_dir, "/tests")
            except Exception as e:
                logger.warning(f"[{self._task_id}] tests upload attempt {attempt + 1} raised: {e}")
                if attempt == 0:
                    continue
                raise
            self._backend.run_command("chmod +x /tests/test.sh 2>/dev/null || true")
            check = self._backend.run_command("test -f /tests/test.sh && echo EXISTS")
            check_stdout = check.stdout.strip()
            if check_stdout == "EXISTS":
                break
            logger.warning(f"[{self._task_id}] /tests/test.sh missing after upload attempt {attempt + 1}; retrying")
        if check_stdout != "EXISTS":
            ls = self._backend.run_command("ls -la /tests 2>&1 || true")
            raise RuntimeError(
                f"No test.sh found in test data for task {self._task_id}. /tests listing: {ls.stdout!r}"
            )

        result = self._backend.run_command(f"timeout {self._test_timeout} bash /tests/test.sh")
        reward = self._parse_reward()

        stdout = _truncate(result.stdout) if result.stdout else ""
        stderr = _truncate(result.stderr) if result.stderr else ""
        observation = (
            f"Test execution complete.\n"
            f"Exit code: {result.exit_code}\n"
            f"[STDOUT]\n{stdout}\n"
            f"[STDERR]\n{stderr}\n"
            f"Reward: {reward}"
        )

        return StepResult(result=observation, reward=reward, done=True)

    def _parse_reward(self) -> float:
        """Parse reward from /logs/verifier/reward.txt. Returns 0.0 if not found."""
        assert self._backend is not None

        try:
            reward_result = self._backend.run_command("cat /logs/verifier/reward.txt")
            if reward_result.exit_code == 0 and reward_result.stdout.strip():
                reward = float(reward_result.stdout.strip())
                return max(0.0, min(1.0, reward))
        except (ValueError, TypeError):
            pass

        return 0.0

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
class SWERLVanilluxSandboxEnvConfig(BaseEnvConfig):
    """Configuration for SWERLVanilluxSandboxEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = SWERLVanilluxSandboxEnv
    backend: str = "docker"
    image: str = "python:3.12-slim"
    mem_limit: str = "4g"
    penalty: float = -0.05
    task_data_dir: str = ""
    task_data_hf_repo: str = ""
    test_timeout: int = 120
    timeout: int = 600
    last_step_warning: bool = False
