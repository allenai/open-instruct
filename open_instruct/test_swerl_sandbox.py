import tempfile
import unittest
from unittest.mock import patch

from open_instruct.environments.backends import ExecutionResult
from open_instruct.environments.base import EnvCall
from open_instruct.environments.swerl_sandbox import LAST_STEP_WARNING, SWERLSandboxEnv


class _FakeBackend:
    def __init__(self):
        self.commands: list[str] = []

    def run_command(self, command: str) -> ExecutionResult:
        self.commands.append(command)
        return ExecutionResult(stdout="ok", stderr="", exit_code=0)


class TestSWERLSandboxTaskDataSetup(unittest.IsolatedAsyncioTestCase):
    async def test_setup_resolves_hf_repo_when_task_data_dir_missing_on_actor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_driver_path = f"{tmpdir}/driver-only-cache"
            actor_path = f"{tmpdir}/actor-cache"
            env = SWERLSandboxEnv(
                task_data_dir=missing_driver_path,
                task_data_hf_repo="hamishivi/swerl-tmax-10k-verified",
            )

            with patch.object(SWERLSandboxEnv, "resolve_task_data_dir", return_value=actor_path) as resolve:
                await env.setup()

            resolve.assert_called_once_with("hamishivi/swerl-tmax-10k-verified")
            self.assertEqual(env._task_data_dir, actor_path)

    async def test_setup_keeps_existing_task_data_dir(self):
        with tempfile.TemporaryDirectory() as task_data_dir:
            env = SWERLSandboxEnv(
                task_data_dir=task_data_dir,
                task_data_hf_repo="hamishivi/swerl-tmax-10k-verified",
            )

            with patch.object(SWERLSandboxEnv, "resolve_task_data_dir") as resolve:
                await env.setup()

            resolve.assert_not_called()
            self.assertEqual(env._task_data_dir, task_data_dir)


class TestSWERLSandboxLastStepWarning(unittest.IsolatedAsyncioTestCase):
    async def test_warning_is_added_to_second_last_observation(self):
        env = SWERLSandboxEnv(last_step_warning=True)
        env._backend = _FakeBackend()
        env._max_steps = 3

        first = await env.step(EnvCall(id="1", name="bash", args={"command": "echo first"}))
        second = await env.step(EnvCall(id="2", name="bash", args={"command": "echo second"}))

        self.assertNotIn(LAST_STEP_WARNING, first.result)
        self.assertIn(LAST_STEP_WARNING, second.result)
