"""SWERL Sandbox environment — per-sample Docker tasks with submit-based evaluation.

Extends GenericSandboxEnv to inherit execute_bash and str_replace_editor tools.
Adds a ``submit`` tool that runs a per-task test script and returns the reward.

Each task has its own files on disk at ``{task_data_dir}/{task_id}/``:
- ``instruction.md`` — task description (returned as observation on reset)
- ``tests/`` — test files copied into the container (``test.sh`` is the entrypoint)
- ``environment/seeds/`` — seed files copied to ``/workspace/``
- ``image.txt`` — (optional) Docker image tag to use for this task
- ``setup.sh`` — (optional) setup script run after seeding
"""

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from open_instruct import logger_utils

from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .generic_sandbox import GenericSandboxEnv

logger = logger_utils.setup_logger(__name__)


_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your solution and run the test suite. Only call when you believe the task is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


class SWERLSandboxEnv(GenericSandboxEnv):
    """Environment for per-sample coding tasks with Docker-based evaluation.

    Extends GenericSandboxEnv to inherit execute_bash and str_replace_editor tools.
    Adds a ``submit`` tool that runs a per-task test script and returns the reward.
    """

    config_name = "swerl_sandbox"
    _tool_definitions = GenericSandboxEnv._tool_definitions + (_SUBMIT_TOOL,)

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
        super().__init__(backend=backend, timeout=timeout, **backend_kwargs)
        self._task_data_dir = task_data_dir
        self._test_timeout = test_timeout
        self._instruction = ""

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        per_sample_config = kwargs.get("env_config") or {}

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

        await super().reset(task_id, **kwargs)

        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)

        assert self._backend is not None

        self._backend.run_command("mkdir -p /workspace")
        self._backend.run_command("echo /workspace > /tmp/.sandbox_cwd")
        self._backend.run_command("mkdir -p /output /logs/verifier")

        self._instruction = ""
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
                    if parent and parent != "/workspace":
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
                    if parent and parent != "/tests":
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

    async def step(self, call: EnvCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name == "submit":
            return self._run_tests()

        return await super().step(call)

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
