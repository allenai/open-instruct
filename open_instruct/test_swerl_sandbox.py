import tempfile
import unittest
from unittest.mock import patch

from open_instruct.environments.swerl_sandbox import SWERLSandboxEnv


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
