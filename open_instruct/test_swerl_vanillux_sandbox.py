import unittest
from unittest.mock import patch

from open_instruct.environments.backends import ExecutionResult
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


class TestSWERLVanilluxSandbox(unittest.IsolatedAsyncioTestCase):
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
            def run_command(self, command: str) -> ExecutionResult:
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

    async def test_bash_output_is_appended_with_exit_code(self):
        env = SWERLVanilluxSandboxEnv()

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertIn("hello", result.result)
        self.assertIn("(exit_code=0)", result.result)

    async def test_bash_output_appends_turns_remaining_when_enabled(self):
        env = SWERLVanilluxSandboxEnv(append_turns_remaining=True)

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()
        env._max_steps = 4

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertTrue(result.result.endswith("(exit_code=0)\nTurns remaining: 3"))

    async def test_bash_output_uses_submit_warning_on_second_last_turn(self):
        env = SWERLVanilluxSandboxEnv(append_turns_remaining=True)

        class _EchoBackend(_FakeBackend):
            def run_command(self, command: str) -> ExecutionResult:
                self.commands.append(command)
                return ExecutionResult(stdout="hello", stderr="", exit_code=0)

        env._backend = _EchoBackend()
        env._max_steps = 2

        result = await env.step(EnvCall(id="1", name="bash", args={"command": "echo hello"}))

        self.assertTrue(result.result.endswith("(exit_code=0)\nOne turn remaining. Please submit your work"))
        self.assertNotIn("Turns remaining: 1", result.result)


if __name__ == "__main__":
    unittest.main()
