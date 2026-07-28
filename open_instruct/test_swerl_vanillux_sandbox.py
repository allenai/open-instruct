import inspect
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from open_instruct.environments.backends import DockerBackend, ExecutionResult, SandboxBackend
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.swerl_vanillux_sandbox import (
    INSTANCE_TEMPLATE,
    SUBMIT_MARKER,
    TOOL_CALL_FORMAT_ERROR_MESSAGE,
    SWERLVanilluxSandboxEnv,
    format_error_message,
    render_instance,
    truncate_observation,
)
from open_instruct.environments.tools.tools import TOOL_REGISTRY

_MODULE = "open_instruct.environments.swerl_vanillux_sandbox"


class _FakeBackend:
    # Keep this signature in step with SandboxBackend.run_command. A fake that omits
    # `timeout` silently accepts calls the real backend would reject with a TypeError.
    def __init__(self):
        self.commands: list[str] = []
        self.timeouts: list[int | None] = []
        self.archives: list[bytes] = []

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        self.commands.append(command)
        self.timeouts.append(timeout)
        return ExecutionResult(stdout="ok", stderr="", exit_code=0)

    def put_archive(self, path: str, data: bytes) -> None:
        self.archives.append(data)

    def write_file(self, path: str, content: str | bytes) -> None:
        self.commands.append(f"write_file {path}")

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        raise FileNotFoundError(path)


class TestSWERLVanilluxSandbox(unittest.IsolatedAsyncioTestCase):
    def test_backend_contract_supports_per_command_timeout(self):
        # This env passes `timeout=` to run_command when running the verifier. The fakes
        # in this file would happily accept it even if the real backends did not, so
        # assert against the actual interface.
        for backend in (SandboxBackend, DockerBackend):
            params = inspect.signature(backend.run_command).parameters
            self.assertIn("timeout", params, f"{backend.__name__}.run_command must accept `timeout`")

    def test_backend_contract_supports_put_archive(self):
        # _upload_directory uploads the test suite via put_archive. Same trap as above:
        # the fakes define it, so only checking the real classes proves it exists.
        for backend in (SandboxBackend, DockerBackend):
            self.assertTrue(hasattr(backend, "put_archive"), f"{backend.__name__} must implement `put_archive`")
        params = inspect.signature(DockerBackend.put_archive).parameters
        self.assertEqual(list(params)[1:], ["path", "data"])

    def test_registered_as_tool_environment(self):
        self.assertIs(TOOL_REGISTRY["swerl_vanillux_sandbox"].tool_class, SWERLVanilluxSandboxEnv)

    def test_tool_surface_is_bash_only(self):
        names = [tool["function"]["name"] for tool in SWERLVanilluxSandboxEnv.get_tool_definitions()]

        self.assertEqual(names, ["bash"])

    def test_tool_call_format_error_message_is_opt_in(self):
        env = SWERLVanilluxSandboxEnv()
        enabled_env = SWERLVanilluxSandboxEnv(tool_call_format_error_feedback=True)

        self.assertIsNone(env.get_tool_call_format_error_message())
        self.assertEqual(enabled_env.get_tool_call_format_error_message(), TOOL_CALL_FORMAT_ERROR_MESSAGE)

    def test_render_instance_substitutes_task(self):
        rendered = render_instance("fix the bug in foo.py")

        self.assertIn("fix the bug in foo.py", rendered)
        self.assertIn("Recommended Workflow", rendered)
        self.assertIn(SUBMIT_MARKER, rendered)
        self.assertIn("{{task}}", INSTANCE_TEMPLATE)
        self.assertNotIn("{{task}}", rendered)

    def test_truncate_observation_keeps_short_outputs(self):
        short = "hello world"

        self.assertEqual(truncate_observation(short), short)

    def test_truncate_observation_applies_head_tail_for_long_outputs(self):
        long_output = ("a" * 6000) + ("b" * 6000)

        truncated = truncate_observation(long_output)

        self.assertIn("HEAD (5000 chars)", truncated)
        self.assertIn("TAIL (5000 chars)", truncated)
        self.assertIn("chars elided", truncated)

    def test_format_error_message_uses_template(self):
        msg = format_error_message("missing tool call")

        self.assertIn("missing tool call", msg)
        self.assertIn("`bash`", msg)
        self.assertIn(SUBMIT_MARKER, msg)

    async def test_unknown_tool_returns_format_error(self):
        env = SWERLVanilluxSandboxEnv()
        env._backend = _FakeBackend()

        result = await env.step(EnvCall(id="1", name="str_replace_editor", args={}))

        self.assertIn("Format error", result.result)
        self.assertIn("`bash`", result.result)
        self.assertFalse(result.done)

    async def test_bash_submit_marker_runs_verifier(self):
        env = SWERLVanilluxSandboxEnv()

        class _SubmitBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                if "wrapper" in command:
                    return ExecutionResult(stdout=SUBMIT_MARKER + "\n", stderr="", exit_code=0)
                return ExecutionResult(stdout="", stderr="", exit_code=0)

        env._backend = _SubmitBackend()
        env._tests_dir = "/tmp/tests"

        with patch.object(env, "_run_tests", return_value=StepResult(result="done", reward=1.0, done=True)) as run:
            result = await env.step(EnvCall(id="1", name="bash", args={"command": f"echo {SUBMIT_MARKER}"}))

        run.assert_called_once_with()
        self.assertEqual(result.result, "done")
        self.assertTrue(result.done)

    def test_run_tests_uploads_runs_and_scores(self):
        # The submit test above patches out _run_tests, so nothing else executes the
        # method that actually uploads tests, runs test.sh and parses the reward —
        # which is the success path of every episode. Exercise it for real here.
        env = SWERLVanilluxSandboxEnv()

        class _VerifierBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                self.timeouts.append(timeout)
                if "test -f /tests/test.sh" in command:
                    return ExecutionResult(stdout="EXISTS\n", stderr="", exit_code=0)
                if "reward.txt" in command:
                    return ExecutionResult(stdout="0.75\n", stderr="", exit_code=0)
                return ExecutionResult(stdout="", stderr="", exit_code=0)

        backend = _VerifierBackend()
        env._backend = backend
        with tempfile.TemporaryDirectory() as tests_dir:
            with open(os.path.join(tests_dir, "test.sh"), "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\nexit 0\n")
            env._tests_dir = tests_dir
            result = env._run_tests()

        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.75)
        self.assertIn("bash /tests/test.sh", backend.commands)
        # The test command must carry the per-command timeout override.
        idx = backend.commands.index("bash /tests/test.sh")
        self.assertEqual(backend.timeouts[idx], env._test_timeout)

    def test_run_tests_reward_is_clamped_and_defaults_to_zero(self):
        env = SWERLVanilluxSandboxEnv()

        class _BadRewardBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                self.timeouts.append(timeout)
                if "test -f /tests/test.sh" in command:
                    return ExecutionResult(stdout="EXISTS\n", stderr="", exit_code=0)
                if "reward.txt" in command:
                    return ExecutionResult(stdout="not-a-number\n", stderr="", exit_code=0)
                return ExecutionResult(stdout="", stderr="", exit_code=0)

        env._backend = _BadRewardBackend()
        with tempfile.TemporaryDirectory() as tests_dir:
            with open(os.path.join(tests_dir, "test.sh"), "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\nexit 0\n")
            env._tests_dir = tests_dir
            result = env._run_tests()

        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.done)

    async def test_bash_output_is_appended_with_exit_code(self):
        env = SWERLVanilluxSandboxEnv()

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertIn("hello", result.result)
        self.assertIn("(exit_code=0)", result.result)

    async def test_bash_output_appends_turns_remaining_when_enabled(self):
        env = SWERLVanilluxSandboxEnv(append_turns_remaining=True)

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()
        env._max_steps = 4

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertTrue(result.result.endswith("(exit_code=0)\nTurns remaining: 3"))

    async def test_bash_output_uses_submit_warning_on_second_last_turn(self):
        env = SWERLVanilluxSandboxEnv(append_turns_remaining=True)

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()
        env._max_steps = 2

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertTrue(result.result.endswith("(exit_code=0)\nOne turn remaining. Please submit your work"))
        self.assertNotIn("Turns remaining: 1", result.result)


if __name__ == "__main__":
    unittest.main()


class TestResolveTaskDataDir(unittest.TestCase):
    """Extraction locking: a dead holder must not wedge everyone else."""

    def _make_repo(self, tmp: str) -> str:
        payload = os.path.join(tmp, "payload")
        os.makedirs(payload, exist_ok=True)
        with open(os.path.join(payload, "test.sh"), "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\nexit 0\n")
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo, exist_ok=True)
        subprocess.run(["tar", "-czf", os.path.join(repo, "task-data.tar.gz"), "-C", payload, "."], check=True)
        return repo

    def test_extracts_tarball_and_marks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp)
            with patch(f"{_MODULE}.snapshot_download", return_value=repo):
                out = SWERLVanilluxSandboxEnv.resolve_task_data_dir("fake/repo")
            self.assertTrue(os.path.isfile(os.path.join(out, ".extraction_complete")))
            self.assertTrue(os.path.isfile(os.path.join(out, "test.sh")))

    def test_second_call_reuses_completed_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp)
            with patch(f"{_MODULE}.snapshot_download", return_value=repo):
                first = SWERLVanilluxSandboxEnv.resolve_task_data_dir("fake/repo")
                # A second call must not re-extract: tar would be invoked again.
                with patch(f"{_MODULE}.subprocess.run") as run:
                    second = SWERLVanilluxSandboxEnv.resolve_task_data_dir("fake/repo")
                run.assert_not_called()
            self.assertEqual(first, second)

    def test_abandoned_lock_file_does_not_block(self):
        # The old implementation used a lock *directory* removed in a finally block,
        # so a killed holder left it behind and every other process span forever.
        # An flock is released by the kernel on process death, so a leftover lock
        # file must not prevent extraction.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(tmp)
            stale_lock = os.path.join(repo, "task-data.tar.gz.extracted.lock")
            with open(stale_lock, "w", encoding="utf-8") as f:
                f.write("")
            with patch(f"{_MODULE}.snapshot_download", return_value=repo):
                out = SWERLVanilluxSandboxEnv.resolve_task_data_dir("fake/repo")
            self.assertTrue(os.path.isfile(os.path.join(out, ".extraction_complete")))

    def test_returns_repo_dir_when_no_tarball(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            os.makedirs(repo, exist_ok=True)
            with patch(f"{_MODULE}.snapshot_download", return_value=repo):
                self.assertEqual(SWERLVanilluxSandboxEnv.resolve_task_data_dir("fake/repo"), repo)
