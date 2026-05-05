import tempfile
import unittest
from unittest.mock import patch

from open_instruct.environments.backends import ExecutionResult
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.swerl_sandbox import LAST_STEP_WARNING, SWERLSandboxEnv
from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv
from open_instruct.environments.tools.tools import TOOL_REGISTRY


class _FakeBackend:
    def __init__(self):
        self.commands: list[str] = []

    def run_command(self, command: str) -> ExecutionResult:
        self.commands.append(command)
        return ExecutionResult(stdout="ok", stderr="", exit_code=0)

    def write_file(self, path: str, content: str | bytes) -> None:
        self.commands.append(f"write_file {path}")

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        raise FileNotFoundError(path)


class _FakeFileBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.files: dict[str, str] = {}

    def run_command(self, command: str) -> ExecutionResult:
        self.commands.append(command)
        if command.startswith("if [ -d "):
            path = command.split("if [ -d ", 1)[1].split(" ];", 1)[0]
            if path in {"/workspace", "/"}:
                stdout = "dir\n"
            elif path in self.files:
                stdout = "file\n"
            else:
                stdout = "missing\n"
            return ExecutionResult(stdout=stdout, stderr="", exit_code=0)
        if command.startswith("test -d "):
            path = command.removeprefix("test -d ")
            return ExecutionResult(stdout="", stderr="", exit_code=0 if path in {"/workspace", "/"} else 1)
        if command.startswith("test -e "):
            path = command.removeprefix("test -e ")
            return ExecutionResult(stdout="", stderr="", exit_code=0 if path in self.files else 1)
        if command.startswith("rm -f "):
            self.files.pop(command.removeprefix("rm -f "), None)
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        return ExecutionResult(stdout="", stderr="", exit_code=0)

    def write_file(self, path: str, content: str | bytes) -> None:
        self.files[path] = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        content = self.files[path]
        return content.encode() if binary else content


class TestSWERLSandboxTaskDataSetup(unittest.IsolatedAsyncioTestCase):
    async def test_setup_resolves_hf_repo_when_task_data_dir_missing_on_actor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_driver_path = f"{tmpdir}/driver-only-cache"
            actor_path = f"{tmpdir}/actor-cache"
            env = SWERLSandboxEnv(
                task_data_dir=missing_driver_path, task_data_hf_repo="hamishivi/swerl-tmax-10k-verified"
            )

            with patch.object(SWERLSandboxEnv, "resolve_task_data_dir", return_value=actor_path) as resolve:
                await env.setup()

            resolve.assert_called_once_with("hamishivi/swerl-tmax-10k-verified")
            self.assertEqual(env._task_data_dir, actor_path)

    async def test_setup_keeps_existing_task_data_dir(self):
        with tempfile.TemporaryDirectory() as task_data_dir:
            env = SWERLSandboxEnv(task_data_dir=task_data_dir, task_data_hf_repo="hamishivi/swerl-tmax-10k-verified")

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


class TestSWERLVanilluxSandbox(unittest.IsolatedAsyncioTestCase):
    def test_registered_as_tool_environment(self):
        self.assertIs(TOOL_REGISTRY["swerl_vanillux_sandbox"].tool_class, SWERLVanilluxSandboxEnv)

    def test_tool_surface_matches_vanillux_setup(self):
        names = [tool["function"]["name"] for tool in SWERLVanilluxSandboxEnv.get_tool_definitions()]

        self.assertEqual(names, ["bash", "str_replace_editor", "submit"])

        editor_tool = SWERLVanilluxSandboxEnv.get_tool_definitions()[1]
        editor_commands = editor_tool["function"]["parameters"]["properties"]["command"]["enum"]
        self.assertEqual(editor_commands, ["view", "create", "str_replace", "insert", "undo_edit"])

        submit_tool = SWERLVanilluxSandboxEnv.get_tool_definitions()[2]
        self.assertEqual(submit_tool["function"]["parameters"]["properties"], {})

    async def test_submit_tool_reviews_then_runs_verifier(self):
        env = SWERLVanilluxSandboxEnv()
        env._backend = _FakeFileBackend()
        env._tests_dir = "/tmp/tests"
        env._write_registry(
            {
                "ROOT": "/app",
                "PROBLEM_STATEMENT": "",
                "SUBMIT_REVIEW_MESSAGES": [
                    "Thank you for your work\n\nHere is a list of all of your changes:\n{{diff}}"
                ],
                "SUBMIT_STAGE": 0,
            }
        )

        with patch.object(env, "_run_tests", return_value=StepResult(result="done", reward=1.0, done=True)) as run:
            review = await env.step(EnvCall(id="1", name="submit", args={}))
            result = await env.step(EnvCall(id="2", name="submit", args={}))

        run.assert_called_once_with()
        self.assertIn("Thank you for your work", review.result)
        self.assertFalse(review.done)
        self.assertEqual(result.result, "done")
        self.assertTrue(result.done)

    async def test_editor_str_replace_and_undo(self):
        env = SWERLVanilluxSandboxEnv()
        backend = _FakeFileBackend()
        backend.files["/workspace/example.txt"] = "hello world\n"
        env._backend = backend

        edit = await env.step(
            EnvCall(
                id="1",
                name="str_replace_editor",
                args={
                    "command": "str_replace",
                    "path": "/workspace/example.txt",
                    "old_str": "world",
                    "new_str": "Vanillux",
                },
            )
        )

        self.assertIn("edited", edit.result)
        self.assertIn("cat -n", edit.result)
        undo = await env.step(
            EnvCall(id="2", name="str_replace_editor", args={"command": "undo_edit", "path": "/workspace/example.txt"})
        )
        self.assertIn("undone", undo.result)
        self.assertEqual(backend.files["/workspace/example.txt"], "hello world\n")
