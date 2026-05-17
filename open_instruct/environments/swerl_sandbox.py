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

import asyncio
import contextlib
import io
import os
import random
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


SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
LAST_STEP_WARNING = (
    "Warning: you only have one more tool call remaining. "
    f"You must end your next tool call with `echo {SUBMIT_MARKER}`"
)

MAX_OUTPUT_CHARS = 10_000
TIMING_LOGS = os.getenv("SWERL_SANDBOX_TIMING_LOGS", "").strip().lower() not in {"", "0", "false", "no", "off"}
TIMING_LOG_THRESHOLD_S = float(os.getenv("SWERL_SANDBOX_TIMING_LOG_THRESHOLD_S", "1.0"))

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


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate keeping first and last half, matching TassieAgent."""
    if len(text) <= limit:
        return text
    half = limit // 2
    n_elided = len(text) - limit
    return f"{text[:half]}\n\n... [{n_elided} characters elided] ...\n\n{text[-half:]}"


class SWERLSandboxEnv(RLEnvironment):
    """Environment for per-sample coding tasks with Docker-based evaluation.

    Provides a single ``bash`` tool. The agent submits by echoing
    ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, which triggers test
    execution and returns the reward.
    """

    config_name = "swerl_sandbox"
    _tool_definitions = (_BASH_TOOL,)
    _RESET_RETRY_BASE_DELAY_S = 1.0
    _RESET_RETRY_MAX_DELAY_S = 16.0
    _RESET_RETRY_JITTER_S = 2.0
    _MIN_TEST_TIMEOUT_S = 600

    def __init__(
        self,
        backend: str = "docker",
        image: str = "python:3.12-slim",
        task_data_dir: str = "",
        task_data_hf_repo: str = "",
        test_timeout: int = 600,
        timeout: int = 120,
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
        self._test_timeout = max(test_timeout, self._MIN_TEST_TIMEOUT_S)
        self._last_step_warning = last_step_warning
        self._max_steps: int | None = None
        self._instruction = ""
        self._tests_dir: str | None = None

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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
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
                    logger.warning(f"SWERLSandboxEnv.reset attempt {attempt + 1} failed: {e}. No attempts remain.")
                    break
                delay = self._reset_retry_delay(attempt + 1)
                logger.warning(
                    "SWERLSandboxEnv.reset attempt %s failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
        if last_error is not None:
            raise RuntimeError(f"Reset failed after {max_attempts} attempts: {last_error}") from last_error
        else:
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

        # Image must be task-specific; no fallback to default constructor image.
        # Rely on sample-level reset kwargs for image selection.
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
                "Missing explicit image for SWERLSandboxEnv reset: %s",
                {k: v for k, v in instance_details.items() if v is not None},
            )
            raise ValueError(
                "SWERLSandboxEnv requires an explicit image per task. "
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
            # Update image in case it changed per-sample
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
        self._max_steps = kwargs.get("max_steps")

        self._backend.run_command("mkdir -p /workspace /output /logs/verifier")
        record_phase("mkdir")

        # Load task data if available
        self._instruction = ""
        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)
        if task_dir and os.path.isdir(task_dir):
            self._load_task_data(task_dir)
        record_phase("load_task_data")

        reset_total_s = time.perf_counter() - reset_start_time
        if TIMING_LOGS and reset_total_s >= TIMING_LOG_THRESHOLD_S:
            logger.info(
                "SWERLSandboxEnv.reset timing task_id=%s image=%s total=%.3fs phases=%s",
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
        """Load instruction, seeds, and setup into the container. Tests are deferred to submit."""
        assert self._backend is not None

        instruction_file = os.path.join(task_dir, "instruction.md")
        if os.path.isfile(instruction_file):
            with open(instruction_file, encoding="utf-8") as f:
                self._instruction = f.read().strip()

        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            self._upload_directory(seeds_dir, "/workspace")

        # Tests are NOT uploaded here — they're uploaded on submit to prevent peeking.
        self._tests_dir = os.path.join(task_dir, "tests") if os.path.isdir(os.path.join(task_dir, "tests")) else None

        # setup.sh is intentionally ignored here. Task images are expected to be
        # prebuilt with dependencies and setup already applied at image build time.

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

        self._backend.put_archive("/", tar_stream.getvalue())

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
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
        else:
            return self._with_last_step_warning(
                StepResult(result=f"Error: Unknown tool '{call.name}'. Available: bash")
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

    # ------------------------------------------------------------------
    # execute_bash
    # ------------------------------------------------------------------
    def _execute_bash(self, args: dict) -> StepResult:
        assert self._backend is not None
        command = args.get("command", "")
        if not command:
            return StepResult(result="Error: 'command' parameter is required.")

        result = self._backend.run_command(command)

        # Build output string exactly like TassieAgent._execute_bash
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}"
        # Prefix with the command to match SFT training data format
        output = f"{command}\n{output}" if output else command

        # Check for submit marker
        if SUBMIT_MARKER in output:
            return self._run_tests()

        output = _truncate(output) or "(no output)"

        return StepResult(result=str(output), metadata={"exit_code": result.exit_code})

    # ------------------------------------------------------------------
    # submit (test execution)
    # ------------------------------------------------------------------
    def _run_tests(self) -> StepResult:
        """Upload tests (deferred from reset to prevent peeking), then run."""
        assert self._backend is not None

        if self._tests_dir is None:
            raise RuntimeError(f"No test data directory for task {self._task_id}")

        # Upload tests only at submit time. The backend silently re-creates
        # the container on NotFound during exec_run, so if the underlying
        # container was recycled between put_archive and the verification
        # exec, /tests won't exist. Retry once to cover that race.
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

        result = self._backend.run_command("bash /tests/test.sh", timeout=self._test_timeout)

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
    test_timeout: int = 600
    timeout: int = 120
    last_step_warning: bool = False
