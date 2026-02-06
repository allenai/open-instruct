"""AgentTask environment — per-sample Docker tasks with submit-based evaluation."""

import os

from open_instruct import logger_utils

from .base import StepResult, ToolCall, register_env
from .sandbox_lm import SandboxLMEnv

logger = logger_utils.setup_logger(__name__)


_SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your solution and run the test suite. Only call when you believe the task is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@register_env("agent_task")
class AgentTaskEnv(SandboxLMEnv):
    """Environment for per-sample coding tasks with Docker-based evaluation.

    Extends SandboxLMEnv to inherit execute_bash and str_replace_editor tools.
    Adds a ``submit`` tool that runs a per-task test script and returns the reward.

    Each task has its own files on disk at ``{task_data_dir}/{task_id}/``:
    - ``instruction.md`` — task description (returned as observation on reset)
    - ``tests/`` — test files copied into the container (``test.sh`` is the entrypoint)
    - ``environment/seeds/`` — seed files copied to ``/workspace/``
    - ``image.txt`` — (optional) Docker image tag to use for this task
    - ``setup.sh`` — (optional) setup script run after seeding
    """

    max_steps = 100

    _tool_definitions = SandboxLMEnv._tool_definitions + [_SUBMIT_TOOL]

    def __init__(
        self,
        backend: str = "docker",
        image: str = "ubuntu:24.04",
        task_data_dir: str = "",
        test_timeout: int = 120,
        timeout: int = 600,
        **backend_kwargs,
    ):
        backend_kwargs["image"] = image
        super().__init__(backend=backend, timeout=timeout, **backend_kwargs)
        self._task_data_dir = task_data_dir
        self._test_timeout = test_timeout
        self._instruction = ""

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        per_sample_config = kwargs.get("env_config") or {}

        # Image priority: env_config["image"] > image.txt on disk > default from __init__
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

        # Parent reset: tears down old container, creates new one, writes bash wrapper
        await super().reset(task_id, **kwargs)

        task_dir = None
        if self._task_data_dir and task_id:
            task_dir = os.path.join(self._task_data_dir, task_id)

        assert self._backend is not None

        # Override cwd to /workspace (instead of parent's /testbed)
        self._backend.run_command("mkdir -p /workspace")
        self._backend.run_command("echo /workspace > /tmp/.sandbox_cwd")

        # Create output/log dirs for test framework
        self._backend.run_command("mkdir -p /output /logs/verifier")

        # Load task data if available
        self._instruction = ""
        if task_dir and os.path.isdir(task_dir):
            self._load_task_data(task_dir)

        observation = self._instruction or "Sandbox ready."
        if not self._instruction and task_id:
            observation = f"[Task: {task_id}] {observation}"

        return StepResult(
            observation=observation,
            tools=self._tool_definitions,
            info={"task_id": task_id, "backend": self._backend_type},
        )

    def _load_task_data(self, task_dir: str) -> None:
        """Load instruction, seeds, and tests from the task directory into the container."""
        assert self._backend is not None

        # Read instruction
        instruction_file = os.path.join(task_dir, "instruction.md")
        if os.path.isfile(instruction_file):
            with open(instruction_file) as f:
                self._instruction = f.read().strip()

        # Copy seed files to /workspace/
        seeds_dir = os.path.join(task_dir, "environment", "seeds")
        if os.path.isdir(seeds_dir):
            for root, _dirs, files in os.walk(seeds_dir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(src_path, seeds_dir)
                    container_path = f"/workspace/{rel_path}"
                    # Ensure parent directory exists
                    parent = os.path.dirname(container_path)
                    if parent and parent != "/workspace":
                        self._backend.run_command(f"mkdir -p {parent}")
                    with open(src_path) as f:
                        content = f.read()
                    self._backend.write_file(container_path, content)

        # Copy test files to /tests/
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
            # Make test.sh executable if it exists
            self._backend.run_command("chmod +x /tests/test.sh 2>/dev/null || true")

        # Run setup.sh if present
        setup_file = os.path.join(task_dir, "setup.sh")
        if os.path.isfile(setup_file):
            with open(setup_file) as f:
                setup_content = f.read()
            self._backend.write_file("/tmp/setup.sh", setup_content)
            self._backend.run_command("chmod +x /tmp/setup.sh && bash /tmp/setup.sh")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    async def step(self, tool_call: ToolCall) -> StepResult:
        if self._backend is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if tool_call.name == "submit":
            return self._run_tests()

        return await super().step(tool_call)

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------
    def _run_tests(self) -> StepResult:
        """Run the task's test script and return the reward."""
        assert self._backend is not None

        # Check if test script exists
        check = self._backend.run_command("test -f /tests/test.sh && echo EXISTS")
        if check.stdout.strip() != "EXISTS":
            return StepResult(observation="No test script found at /tests/test.sh. Reward: 0.0", reward=0.0, done=True)

        # Run the test script with timeout
        result = self._backend.run_command(f"timeout {self._test_timeout} bash /tests/test.sh")

        # Try to read reward from /logs/verifier/reward.txt
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

        return StepResult(observation=observation, reward=reward, done=True)

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

        # Fallback: exit code 0 → 1.0, nonzero → 0.0
        return 1.0 if exit_code == 0 else 0.0
