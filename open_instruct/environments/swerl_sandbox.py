"""SWERL Sandbox environment — per-sample Docker tasks with bash-only tool loop.

Provides a single ``bash`` tool inside a Docker container. The agent submits
by echoing ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``.

Each task has its own files on disk at ``{task_data_dir}/{task_id}/``:
- ``instruction.md`` — task description (returned as observation on reset)
- ``tests/`` — test files copied into the container (``test.sh`` is the entrypoint)
- ``environment/seeds/`` — seed files copied to ``/workspace/``
- ``image.txt`` — (optional) Docker image tag to use for this task
- ``setup.sh`` — (optional) setup script run after seeding
"""

import contextlib
import io
import os
import shlex
import tarfile
from dataclasses import dataclass
from typing import Any, ClassVar

from huggingface_hub import snapshot_download
from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .backends import SandboxBackend, create_backend
from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .tools.utils import coerce_args

logger = logger_utils.setup_logger(__name__)


_BASH_WRAPPER = r"""#!/bin/bash
set -a; source /tmp/.sandbox_env 2>/dev/null; set +a
cd "$(cat /tmp/.sandbox_cwd 2>/dev/null || echo /workspace)" 2>/dev/null
eval "$1"
_exit_code=$?
export -p > /tmp/.sandbox_env
pwd > /tmp/.sandbox_cwd
exit $_exit_code
"""

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Each command runs in a new subshell.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
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


class SWERLSandboxEnv(RLEnvironment):
    """Environment for per-sample coding tasks with Docker-based evaluation.

    Provides a single ``bash`` tool. The agent submits by echoing
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, which triggers test
    execution and returns the reward.
    """

    config_name = "swerl_sandbox"
    _tool_definitions = (_BASH_TOOL,)

    def __init__(
        self,
        backend: str = "docker",
        image: str = "python:3.12-slim",
        task_data_dir: str = "",
        task_data_hf_repo: str = "",
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
        self._task_data_hf_repo = task_data_hf_repo
        self._test_timeout = test_timeout
        self._instruction = ""

    async def setup(self) -> None:
        """Download task data from HuggingFace if configured."""
        if self._task_data_hf_repo and not self._task_data_dir:
            logger.info(f"Downloading task data from {self._task_data_hf_repo}...")
            self._task_data_dir = snapshot_download(self._task_data_hf_repo, repo_type="dataset")
            logger.info(f"Task data cached at {self._task_data_dir}")

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
                with open(image_file, encoding="utf-8") as f:
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
            with open(instruction_file, encoding="utf-8") as f:
                self._instruction = f.read().strip()

        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            self._upload_directory(seeds_dir, "/workspace")

        tests_dir = os.path.join(task_dir, "tests")
        if os.path.isdir(tests_dir):
            self._upload_directory(tests_dir, "/tests")
            self._backend.run_command("chmod +x /tests/test.sh 2>/dev/null || true")

        setup_file = os.path.join(task_dir, "setup.sh")
        if os.path.isfile(setup_file):
            with open(setup_file, encoding="utf-8") as f:
                setup_content = f.read()
            self._backend.write_file("/tmp/setup.sh", setup_content)
            self._backend.run_command("chmod +x /tmp/setup.sh && bash /tmp/setup.sh")

    def _upload_directory(self, host_dir: str, container_dir: str) -> None:
        """Upload a local directory tree into the container as a single tar archive.

        Builds an in-memory tar with full container paths (e.g. /tests/test.sh)
        and extracts at / so no pre-existing directories are required.
        """
        assert self._backend is not None
        # Strip leading slash for tar entry names (put_archive at "/" adds it back)
        prefix = container_dir.lstrip("/")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            # Add the root directory entry
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

        tar_stream.seek(0)
        self._backend._container.put_archive("/", tar_stream)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    async def step(self, call: EnvCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name in ("bash", "execute_bash"):
            args = coerce_args(_BASH_TOOL["function"]["parameters"], call.args)
            return self._execute_bash(args)
        else:
            return StepResult(result=f"Error: Unknown tool '{call.name}'. Available: bash", reward=self._penalty)

    # ------------------------------------------------------------------
    # execute_bash
    # ------------------------------------------------------------------
    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        if not command:
            return StepResult(result="Error: 'command' parameter is required.", reward=self._penalty)

        result = self._backend.run_command(f"bash /tmp/.sandbox_bash_wrapper.sh {shlex.quote(command)}")

        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.exit_code != 0:
            output = f"Exit code {result.exit_code}\n{output}"

        output = _truncate_output(output) if output else "(no output)"

        # Check for submit marker
        if SUBMIT_MARKER in (result.stdout or ""):
            return self._run_tests()

        return StepResult(result=output, metadata={"exit_code": result.exit_code})

    # ------------------------------------------------------------------
    # submit (test execution)
    # ------------------------------------------------------------------
    def _run_tests(self) -> StepResult:
        """Run the task's test script and return the reward."""
        assert self._backend is not None

        check = self._backend.run_command("test -f /tests/test.sh && echo EXISTS")
        if check.stdout.strip() != "EXISTS":
            return StepResult(result="No test script found at /tests/test.sh. Reward: 0.2", reward=0.2, done=True)

        result = self._backend.run_command(f"timeout {self._test_timeout} bash /tests/test.sh")

        reward = self._parse_reward()

        stdout = _truncate_output(result.stdout) if result.stdout else ""
        stderr = _truncate_output(result.stderr) if result.stderr else ""

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
    task_data_hf_repo: str = ""
    test_timeout: int = 120
    timeout: int = 600
