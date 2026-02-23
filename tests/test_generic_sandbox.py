"""Unit tests for GenericSandboxEnv with a mock Docker backend."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from open_instruct.environments.backends import ExecutionResult, SandboxBackend
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.generic_sandbox import GenericSandboxEnv


class MockBackend(SandboxBackend):
    """In-memory mock backend that simulates a filesystem and command execution."""

    def __init__(self, **_kwargs):
        self._files: dict[str, str | bytes] = {}
        self._cwd = "/testbed"
        self._started = False

    def start(self) -> None:
        self._started = True
        self._files = {}
        self._cwd = "/testbed"

    def run_command(self, command: str) -> ExecutionResult:
        if "which git" in command:
            return ExecutionResult(stdout="", stderr="", exit_code=1)
        if "mkdir -p" in command:
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        if "chmod" in command:
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        if "echo /testbed" in command:
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        if "test -d" in command:
            path = command.split("test -d ")[-1].strip().strip("'\"")
            if path in self._files and self._files[path] == "__DIR__":
                return ExecutionResult(stdout="", stderr="", exit_code=0)
            return ExecutionResult(stdout="", stderr="", exit_code=1)
        if "test -e" in command:
            path = command.split("test -e ")[-1].strip().strip("'\"")
            if path in self._files:
                return ExecutionResult(stdout="", stderr="", exit_code=0)
            return ExecutionResult(stdout="", stderr="", exit_code=1)
        if "cat -n" in command:
            parts = command.split("cat -n ")[-1]
            path = parts.split("|")[0].strip().strip("'\"")
            if path in self._files:
                content = self._files[path]
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                lines = content.splitlines()
                numbered = "\n".join(f"     {i + 1}\t{line}" for i, line in enumerate(lines))
                return ExecutionResult(stdout=numbered, stderr="", exit_code=0)
            return ExecutionResult(stdout="", stderr=f"cat: {path}: No such file", exit_code=1)
        if "find " in command:
            return ExecutionResult(stdout="/testbed\n/testbed/input\n/testbed/output", stderr="", exit_code=0)
        if "sandbox_bash_wrapper" in command:
            cmd_part = command.split("bash /tmp/.sandbox_bash_wrapper.sh ")[-1].strip("'\"")
            if "echo hello" in cmd_part:
                return ExecutionResult(stdout="hello\n", stderr="", exit_code=0)
            if "exit 1" in cmd_part:
                return ExecutionResult(stdout="", stderr="error", exit_code=1)
            if "ls /testbed" in cmd_part:
                return ExecutionResult(stdout="input\noutput\n", stderr="", exit_code=0)
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        return ExecutionResult(stdout="", stderr="", exit_code=0)

    def write_file(self, path: str, content: str | bytes) -> None:
        self._files[path] = content

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        if path not in self._files:
            raise FileNotFoundError(f"File not found: '{path}'")
        content = self._files[path]
        if binary:
            return content if isinstance(content, bytes) else content.encode("utf-8")
        return content if isinstance(content, str) else content.decode("utf-8", errors="replace")

    def close(self) -> None:
        self._started = False


_MOCK_BACKEND_PATCH = "open_instruct.environments.generic_sandbox.create_backend"


def _mock_create_backend(_backend_type="docker", **_kw):
    return MockBackend()


def _make_env(**kwargs) -> GenericSandboxEnv:
    """Create a GenericSandboxEnv that uses MockBackend instead of Docker."""
    with patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend):
        env = GenericSandboxEnv(backend="docker", **kwargs)
        asyncio.run(env.reset(task_id="test-task"))
    return env


def _step(env: GenericSandboxEnv, name: str, args: dict) -> StepResult:
    """Helper to run a step synchronously."""
    call = EnvCall(id="test", name=name, args=args)
    return asyncio.run(env.step(call))


class TestGenericSandboxReset(unittest.TestCase):

    @patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend)
    def test_reset_returns_observation_and_tools(self, _mock):
        env = GenericSandboxEnv(backend="docker")
        result, tools = asyncio.run(env.reset(task_id="my-task"))

        self.assertIsInstance(result, StepResult)
        self.assertIn("Sandbox ready", result.result)
        self.assertIn("my-task", result.result)
        self.assertEqual(len(tools), 2)
        tool_names = {t["function"]["name"] for t in tools}
        self.assertEqual(tool_names, {"execute_bash", "str_replace_editor"})

    @patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend)
    def test_reset_without_task_id(self, _mock):
        env = GenericSandboxEnv(backend="docker")
        result, tools = asyncio.run(env.reset())

        self.assertIn("Sandbox ready", result.result)
        self.assertNotIn("[Task:", result.result)

    @patch(_MOCK_BACKEND_PATCH)
    def test_reset_retries_on_failure(self, mock_create):
        call_count = 0

        def create_side_effect(_backend_type="docker", **_kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                b = MagicMock()
                b.start.side_effect = RuntimeError("Docker not running")
                return b
            return MockBackend()

        mock_create.side_effect = create_side_effect

        env = GenericSandboxEnv(backend="docker")
        result, tools = asyncio.run(env.reset(task_id="retry-test"))
        self.assertIn("Sandbox ready", result.result)
        self.assertEqual(call_count, 3)


class TestExecuteBash(unittest.TestCase):

    def test_successful_command(self):
        env = _make_env()
        result = _step(env, "execute_bash", {"command": "echo hello"})

        self.assertIn("hello", result.result)
        self.assertIn("Exit code: 0", result.result)
        self.assertEqual(result.reward, 0.0)

    def test_failed_command(self):
        env = _make_env()
        result = _step(env, "execute_bash", {"command": "exit 1"})

        self.assertIn("Exit code: 1", result.result)
        self.assertEqual(result.reward, -0.05)

    def test_empty_command(self):
        env = _make_env()
        result = _step(env, "execute_bash", {"command": ""})

        self.assertIn("Error", result.result)
        self.assertEqual(result.reward, -0.05)

    def test_missing_command(self):
        env = _make_env()
        result = _step(env, "execute_bash", {})

        self.assertIn("'command' parameter is required", result.result)
        self.assertEqual(result.reward, -0.05)


class TestStrReplaceEditor(unittest.TestCase):

    def test_create_file(self):
        env = _make_env()
        result = _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/hello.py",
            "file_text": "print('hello')\n",
        })

        self.assertIn("File created successfully", result.result)
        self.assertEqual(result.reward, 0.0)

    def test_create_existing_file_errors(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/exists.py",
            "file_text": "original",
        })
        result = _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/exists.py",
            "file_text": "overwrite",
        })

        self.assertIn("already exists", result.result)
        self.assertIn("ERROR", result.result)

    def test_create_without_file_text(self):
        env = _make_env()
        result = _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/no_content.py",
        })

        self.assertIn("'file_text' parameter is required", result.result)

    def test_view_file(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/view_me.txt",
            "file_text": "line one\nline two\nline three\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "view",
            "path": "/testbed/view_me.txt",
        })

        self.assertIn("str_replace_editor", result.result)
        self.assertIn("line one", result.result)

    def test_view_nonexistent_file(self):
        env = _make_env()
        result = _step(env, "str_replace_editor", {
            "command": "view",
            "path": "/testbed/nope.txt",
        })

        self.assertIn("ERROR", result.result)

    def test_str_replace(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/edit.py",
            "file_text": "x = 1\ny = 2\nz = 3\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "str_replace",
            "path": "/testbed/edit.py",
            "old_str": "y = 2",
            "new_str": "y = 42",
        })

        self.assertIn("has been edited", result.result)

    def test_str_replace_not_found(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/sr.py",
            "file_text": "a = 1\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "str_replace",
            "path": "/testbed/sr.py",
            "old_str": "not here",
            "new_str": "replacement",
        })

        self.assertIn("old_str not found", result.result)

    def test_str_replace_multiple_matches(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/dup.py",
            "file_text": "x = 1\nx = 1\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "str_replace",
            "path": "/testbed/dup.py",
            "old_str": "x = 1",
            "new_str": "x = 2",
        })

        self.assertIn("found 2 times", result.result)

    def test_insert(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/ins.py",
            "file_text": "line 1\nline 3\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "insert",
            "path": "/testbed/ins.py",
            "insert_line": 1,
            "new_str": "line 2",
        })

        self.assertIn("has been edited", result.result)

    def test_insert_missing_params(self):
        env = _make_env()
        _step(env, "str_replace_editor", {
            "command": "create",
            "path": "/testbed/ins2.py",
            "file_text": "content\n",
        })
        result = _step(env, "str_replace_editor", {
            "command": "insert",
            "path": "/testbed/ins2.py",
        })

        self.assertIn("'insert_line' parameter is required", result.result)


class TestUnknownTool(unittest.TestCase):

    def test_unknown_tool_name(self):
        env = _make_env()
        result = _step(env, "nonexistent_tool", {"arg": "val"})

        self.assertIn("Unknown tool", result.result)
        self.assertEqual(result.reward, -0.05)


class TestMetricsAndState(unittest.TestCase):

    def test_step_count_increments(self):
        env = _make_env()
        self.assertEqual(env.get_metrics()["step_count"], 0.0)

        _step(env, "execute_bash", {"command": "echo hello"})
        self.assertEqual(env.get_metrics()["step_count"], 1.0)

        _step(env, "execute_bash", {"command": "echo world"})
        self.assertEqual(env.get_metrics()["step_count"], 2.0)

    def test_state_returns_task_id(self):
        env = _make_env()
        state = env.state()
        self.assertEqual(state.episode_id, "test-task")


class TestConfigurablePenalty(unittest.TestCase):

    @patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend)
    def test_custom_penalty(self, _mock):
        env = GenericSandboxEnv(backend="docker", penalty=-0.1)
        asyncio.run(env.reset(task_id="penalty-test"))

        result = _step(env, "execute_bash", {"command": "exit 1"})
        self.assertEqual(result.reward, -0.1)

        result = _step(env, "nonexistent_tool", {})
        self.assertEqual(result.reward, -0.1)


class TestCloseAndShutdown(unittest.TestCase):

    @patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend)
    def test_close(self, _mock):
        env = GenericSandboxEnv(backend="docker")
        asyncio.run(env.reset(task_id="close-test"))
        asyncio.run(env.close())
        self.assertIsNone(env._backend)

    @patch(_MOCK_BACKEND_PATCH, side_effect=_mock_create_backend)
    def test_shutdown(self, _mock):
        env = GenericSandboxEnv(backend="docker")
        asyncio.run(env.reset(task_id="shutdown-test"))
        asyncio.run(env.shutdown())
        self.assertIsNone(env._backend)
