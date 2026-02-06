"""Unit tests for RL environments."""

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

from open_instruct.environments import ENV_REGISTRY, EnvironmentState, StepResult, ToolCall, get_env_class
from open_instruct.environments.agent_task import AgentTaskEnv
from open_instruct.environments.backends import DaytonaBackend, ExecutionResult, create_backend
from open_instruct.environments.base import RLEnvironment
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv
from open_instruct.environments.sandbox_lm import SandboxLMEnv, _truncate_output
from open_instruct.ground_truth_utils import LastRewardAggregator, SumRewardAggregator
from open_instruct.tools.utils import Tool


def run_async(coro):
    """Run async function in sync test."""
    return asyncio.run(coro)


class TestDataClasses:
    """Test core data classes."""

    def test_tool_call(self):
        tc = ToolCall(name="test", args={"x": 1})
        assert tc.name == "test"
        assert tc.args == {"x": 1}

    def test_environment_state(self):
        state = EnvironmentState(rewards=[0.1, 0.2, 0.5])
        assert state.final_reward == 0.5
        assert abs(state.total_reward - 0.8) < 0.001

    def test_empty_state(self):
        state = EnvironmentState()
        assert state.final_reward == 0.0
        assert state.total_reward == 0.0


class TestInheritance:
    """Test that RLEnvironment extends Tool."""

    def test_rlenvironment_is_tool(self):
        assert issubclass(RLEnvironment, Tool)

    def test_counter_env_is_tool(self):
        assert issubclass(CounterEnv, Tool)


class TestRegistry:
    """Test environment registry."""

    def test_envs_registered(self):
        assert "counter" in ENV_REGISTRY
        assert "sandbox" in ENV_REGISTRY

    def test_get_env_class(self):
        cls = get_env_class(env_name="counter")
        assert cls == CounterEnv


class TestCounterEnv:
    """Test CounterEnv."""

    def test_full_episode(self):
        async def _test():
            env = CounterEnv(target=3)
            result = await env.reset()
            assert isinstance(result, StepResult)
            assert result.tools is not None
            assert len(result.tools) == 3

            for _ in range(3):
                await env.step(ToolCall(name="increment", args={}))

            step = await env.step(ToolCall(name="submit", args={}))
            assert step.done
            assert step.reward == 1.0

        run_async(_test())


class TestGuessNumberEnv:
    """Test GuessNumberEnv."""

    def test_correct_guess(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="5")
            result = await env.step(ToolCall(name="guess", args={"number": 5}))
            assert result.done
            assert result.reward == 1.0

        run_async(_test())


class TestRewardAggregators:
    """Test reward aggregators (replaced env verifiers)."""

    def test_last_reward_aggregator(self):
        agg = LastRewardAggregator()
        assert agg([0.1, 0.5, 1.0]) == 1.0

    def test_last_reward_aggregator_empty(self):
        agg = LastRewardAggregator()
        assert agg([]) == 0.0

    def test_sum_reward_aggregator(self):
        agg = SumRewardAggregator()
        assert agg([1.0, 2.0, 3.0]) == 6.0

    def test_sum_reward_aggregator_empty(self):
        agg = SumRewardAggregator()
        assert agg([]) == 0.0


# ---------------------------------------------------------------------------
# SandboxLMEnv tests (mocked backends)
# ---------------------------------------------------------------------------
def _make_mock_backend():
    """Create a mock SandboxBackend with sensible defaults."""
    backend = MagicMock()
    backend.run_command.return_value = ExecutionResult(stdout="", stderr="", exit_code=0)
    backend.read_file.return_value = ""
    return backend


class TestSandboxLMEnv:
    """Tests for the sandbox_lm environment with mocked backends."""

    def test_registration(self):
        assert "sandbox_lm" in ENV_REGISTRY

    def test_tool_definitions(self):
        env = SandboxLMEnv()
        tools = env.get_tool_definitions()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"execute_bash", "str_replace_editor"}

    def test_reset(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv(backend="docker")
                result = await env.reset(task_id="test-task-1")

            assert isinstance(result, StepResult)
            assert result.tools is not None
            assert len(result.tools) == 2
            assert "test-task-1" in result.observation
            mock_backend.start.assert_called_once()
            # Verify setup commands were run
            commands = [call.args[0] for call in mock_backend.run_command.call_args_list]
            assert any("mkdir" in c for c in commands)
            assert any("git init" in c for c in commands)

        run_async(_test())

    def test_execute_bash(self):
        async def _test():
            mock_backend = _make_mock_backend()
            mock_backend.run_command.return_value = ExecutionResult(
                stdout="hello world", stderr="", exit_code=0
            )
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            # Now execute bash
            result = await env.step(ToolCall(name="execute_bash", args={"command": "echo hello world"}))
            assert result.reward == 0.0
            assert "hello world" in result.observation
            assert "Exit code: 0" in result.observation

        run_async(_test())

    def test_execute_bash_nonzero_exit(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            mock_backend.run_command.return_value = ExecutionResult(
                stdout="", stderr="not found", exit_code=1
            )
            result = await env.step(ToolCall(name="execute_bash", args={"command": "bad_cmd"}))
            assert result.reward == -0.05
            assert "Exit code: 1" in result.observation

        run_async(_test())

    def test_editor_view(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            # First call checks if path is dir, second call is cat -n
            mock_backend.run_command.side_effect = [
                ExecutionResult(stdout="FILE", stderr="", exit_code=0),
                ExecutionResult(stdout="     1\thello\n     2\tworld\n", stderr="", exit_code=0),
            ]
            result = await env.step(
                ToolCall(name="str_replace_editor", args={"command": "view", "path": "/testbed/foo.py"})
            )
            assert result.reward == 0.0
            assert "hello" in result.observation

        run_async(_test())

    def test_editor_create(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            # test -e check returns empty (does not exist), then mkdir, then write_file
            mock_backend.run_command.side_effect = [
                ExecutionResult(stdout="", stderr="", exit_code=1),  # test -e
                ExecutionResult(stdout="", stderr="", exit_code=0),  # mkdir -p
            ]
            result = await env.step(
                ToolCall(
                    name="str_replace_editor",
                    args={"command": "create", "path": "/testbed/new.py", "file_text": "print('hi')"},
                )
            )
            assert result.reward == 0.0
            assert "File created successfully" in result.observation
            mock_backend.write_file.assert_called_with("/testbed/new.py", "print('hi')")

        run_async(_test())

    def test_editor_create_fails_if_exists(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            mock_backend.run_command.return_value = ExecutionResult(stdout="EXISTS", stderr="", exit_code=0)
            result = await env.step(
                ToolCall(
                    name="str_replace_editor",
                    args={"command": "create", "path": "/testbed/existing.py", "file_text": "x"},
                )
            )
            assert result.reward == -0.05
            assert "already exists" in result.observation

        run_async(_test())

    def test_editor_str_replace(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            mock_backend.read_file.return_value = "hello world\ngoodbye world\n"
            result = await env.step(
                ToolCall(
                    name="str_replace_editor",
                    args={
                        "command": "str_replace",
                        "path": "/testbed/foo.py",
                        "old_str": "hello world",
                        "new_str": "hi world",
                    },
                )
            )
            assert result.reward == 0.0
            assert "has been edited" in result.observation
            mock_backend.write_file.assert_called_with("/testbed/foo.py", "hi world\ngoodbye world\n")

        run_async(_test())

    def test_editor_str_replace_multiple_matches(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            mock_backend.read_file.return_value = "aaa\naaa\naaa\n"
            result = await env.step(
                ToolCall(
                    name="str_replace_editor",
                    args={
                        "command": "str_replace",
                        "path": "/testbed/foo.py",
                        "old_str": "aaa",
                        "new_str": "bbb",
                    },
                )
            )
            assert result.reward == -0.05
            assert "3 times" in result.observation

        run_async(_test())

    def test_editor_insert(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            mock_backend.read_file.return_value = "line1\nline2\nline3\n"
            result = await env.step(
                ToolCall(
                    name="str_replace_editor",
                    args={
                        "command": "insert",
                        "path": "/testbed/foo.py",
                        "insert_line": 1,
                        "new_str": "inserted",
                    },
                )
            )
            assert result.reward == 0.0
            assert "has been edited" in result.observation
            written = mock_backend.write_file.call_args[0][1]
            assert "inserted" in written
            lines = written.splitlines()
            assert lines[0] == "line1"
            assert lines[1] == "inserted"
            assert lines[2] == "line2"

        run_async(_test())

    def test_write_prompt_file(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv(write_prompt_file=True)
                await env.reset(task_id="my-task-prompt")

            # write_file should have been called for the wrapper AND prompt.txt
            write_calls = mock_backend.write_file.call_args_list
            prompt_calls = [c for c in write_calls if c.args[0] == "/root/prompt.txt"]
            assert len(prompt_calls) == 1
            assert prompt_calls[0].args[1] == "my-task-prompt"

        run_async(_test())

    def test_unknown_tool(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = SandboxLMEnv()
                await env.reset()

            result = await env.step(ToolCall(name="nonexistent", args={}))
            assert result.reward == -0.05
            assert "Unknown tool" in result.observation

        run_async(_test())


class TestTruncateOutput:
    """Test the _truncate_output helper."""

    def test_short_text_unchanged(self):
        text = "\n".join(f"line {i}" for i in range(10))
        assert _truncate_output(text) == text

    def test_long_text_truncated(self):
        text = "\n".join(f"line {i}" for i in range(200))
        result = _truncate_output(text, num_lines=40)
        assert "line 0" in result
        assert "line 199" in result
        assert "<Observation truncated in middle for saving context>" in result


# ---------------------------------------------------------------------------
# AgentTaskEnv tests (mocked backends + temp directories)
# ---------------------------------------------------------------------------
class TestAgentTaskEnv:
    """Tests for the agent_task environment with mocked backends."""

    def test_registration(self):
        assert "agent_task" in ENV_REGISTRY

    def test_tool_definitions(self):
        env = AgentTaskEnv()
        tools = env.get_tool_definitions()
        assert len(tools) == 3
        names = {t["function"]["name"] for t in tools}
        assert names == {"execute_bash", "str_replace_editor", "submit"}

    def test_reset_with_task_data(self):
        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create task data structure
                task_id = "task_001"
                task_dir = os.path.join(tmpdir, task_id)
                os.makedirs(os.path.join(task_dir, "environment", "seeds"))
                os.makedirs(os.path.join(task_dir, "tests"))

                # Write instruction
                with open(os.path.join(task_dir, "instruction.md"), "w") as f:
                    f.write("Fix the bug in main.py")

                # Write seed file
                with open(os.path.join(task_dir, "environment", "seeds", "main.py"), "w") as f:
                    f.write("print('hello')")

                # Write test script
                with open(os.path.join(task_dir, "tests", "test.sh"), "w") as f:
                    f.write("#!/bin/bash\npython main.py")

                mock_backend = _make_mock_backend()
                with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                    env = AgentTaskEnv(task_data_dir=tmpdir)
                    result = await env.reset(task_id=task_id)

                assert isinstance(result, StepResult)
                assert "Fix the bug in main.py" in result.observation
                assert result.tools is not None
                assert len(result.tools) == 3

                # Verify seeds were copied
                write_calls = mock_backend.write_file.call_args_list
                seed_calls = [c for c in write_calls if "/workspace/main.py" in c.args[0]]
                assert len(seed_calls) == 1
                assert seed_calls[0].args[1] == "print('hello')"

                # Verify test script was copied
                test_calls = [c for c in write_calls if "/tests/test.sh" in c.args[0]]
                assert len(test_calls) == 1

        run_async(_test())

    def test_reset_without_task_data(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                result = await env.reset()

            assert isinstance(result, StepResult)
            assert "Sandbox ready." in result.observation

        run_async(_test())

    def test_submit_success(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                await env.reset()

            # test -f check → EXISTS, run test.sh → success, cat reward.txt → "1"
            mock_backend.run_command.side_effect = [
                ExecutionResult(stdout="EXISTS", stderr="", exit_code=0),  # test -f
                ExecutionResult(stdout="All tests passed", stderr="", exit_code=0),  # test.sh
                ExecutionResult(stdout="1", stderr="", exit_code=0),  # cat reward.txt
            ]
            result = await env.step(ToolCall(name="submit", args={}))
            assert result.done is True
            assert result.reward == 1.0

        run_async(_test())

    def test_submit_failure(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                await env.reset()

            # test -f check → EXISTS, run test.sh → failure, cat reward.txt → "0"
            mock_backend.run_command.side_effect = [
                ExecutionResult(stdout="EXISTS", stderr="", exit_code=0),  # test -f
                ExecutionResult(stdout="FAILED", stderr="Error", exit_code=1),  # test.sh
                ExecutionResult(stdout="0", stderr="", exit_code=0),  # cat reward.txt
            ]
            result = await env.step(ToolCall(name="submit", args={}))
            assert result.done is True
            assert result.reward == 0.0

        run_async(_test())

    def test_submit_no_test_script(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                await env.reset()

            # test -f check → not EXISTS
            mock_backend.run_command.return_value = ExecutionResult(stdout="", stderr="", exit_code=1)
            result = await env.step(ToolCall(name="submit", args={}))
            assert result.done is True
            assert result.reward == 0.0
            assert "No test script" in result.observation

        run_async(_test())

    def test_bash_inherited(self):
        async def _test():
            mock_backend = _make_mock_backend()
            mock_backend.run_command.return_value = ExecutionResult(
                stdout="hello from agent_task", stderr="", exit_code=0
            )
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                await env.reset()

            result = await env.step(ToolCall(name="execute_bash", args={"command": "echo hello"}))
            assert result.reward == 0.0
            assert "hello from agent_task" in result.observation

        run_async(_test())

    def test_workspace_cwd(self):
        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv()
                await env.reset()

            # Verify /workspace cwd setup commands ran
            commands = [call.args[0] for call in mock_backend.run_command.call_args_list]
            assert any("mkdir -p /workspace" in c for c in commands)
            assert any("/workspace" in c and ".sandbox_cwd" in c for c in commands)

        run_async(_test())

    def test_reset_image_from_env_config(self):
        """env_config['image'] should override the default image."""

        async def _test():
            mock_backend = _make_mock_backend()
            with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                env = AgentTaskEnv(image="ubuntu:24.04")
                await env.reset(task_id="task_001", env_config={"task_id": "task_001", "image": "custom:latest"})

            # The backend should have been created with the env_config image
            assert env._backend_kwargs["image"] == "custom:latest"

        run_async(_test())

    def test_reset_image_fallback_to_file(self):
        """image.txt on disk should be used when env_config has no image."""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                task_id = "task_img"
                task_dir = os.path.join(tmpdir, task_id)
                os.makedirs(task_dir)
                with open(os.path.join(task_dir, "image.txt"), "w") as f:
                    f.write("from-file:v1")

                mock_backend = _make_mock_backend()
                with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                    env = AgentTaskEnv(task_data_dir=tmpdir, image="default:img")
                    await env.reset(task_id=task_id, env_config={"task_id": task_id})

                assert env._backend_kwargs["image"] == "from-file:v1"

        run_async(_test())

    def test_reset_env_config_image_overrides_file(self):
        """env_config['image'] should take priority over image.txt on disk."""

        async def _test():
            with tempfile.TemporaryDirectory() as tmpdir:
                task_id = "task_both"
                task_dir = os.path.join(tmpdir, task_id)
                os.makedirs(task_dir)
                with open(os.path.join(task_dir, "image.txt"), "w") as f:
                    f.write("from-file:v1")

                mock_backend = _make_mock_backend()
                with patch("open_instruct.environments.sandbox_lm.create_backend", return_value=mock_backend):
                    env = AgentTaskEnv(task_data_dir=tmpdir, image="default:img")
                    await env.reset(
                        task_id=task_id, env_config={"task_id": task_id, "image": "from-config:v2"}
                    )

                # env_config image should win over image.txt
                assert env._backend_kwargs["image"] == "from-config:v2"

        run_async(_test())


# ---------------------------------------------------------------------------
# DaytonaBackend tests (mocked SDK)
# ---------------------------------------------------------------------------
class TestDaytonaBackend:
    """Tests for the Daytona sandbox backend with mocked SDK."""

    def test_create_backend_daytona(self):
        backend = create_backend("daytona", image="python:3.12")
        assert isinstance(backend, DaytonaBackend)
        assert backend._image == "python:3.12"

    @patch("open_instruct.environments.backends.HAS_DAYTONA", True)
    @patch("open_instruct.environments.backends.Daytona")
    @patch("open_instruct.environments.backends.DaytonaConfig")
    def test_start(self, mock_config_cls, mock_daytona_cls):
        mock_client = MagicMock()
        mock_sandbox = MagicMock()
        mock_daytona_cls.return_value = mock_client
        mock_client.create.return_value = mock_sandbox

        backend = DaytonaBackend(image="python:3.12", api_key="test-key")
        backend.start()

        mock_config_cls.assert_called_once_with(api_key="test-key")
        mock_daytona_cls.assert_called_once_with(mock_config_cls.return_value)
        mock_client.create.assert_called_once_with(image="python:3.12")
        assert backend._sandbox is mock_sandbox

    @patch("open_instruct.environments.backends.HAS_DAYTONA", True)
    @patch("open_instruct.environments.backends.Daytona")
    def test_start_no_config(self, mock_daytona_cls):
        mock_client = MagicMock()
        mock_daytona_cls.return_value = mock_client
        mock_client.create.return_value = MagicMock()

        backend = DaytonaBackend()
        backend.start()

        # No config args → Daytona() called with no arguments
        mock_daytona_cls.assert_called_once_with()

    def test_start_import_error(self):
        with patch("open_instruct.environments.backends.HAS_DAYTONA", False):
            backend = DaytonaBackend()
            try:
                backend.start()
                assert False, "Expected ImportError"
            except ImportError:
                pass

    def test_run_command(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        mock_response = MagicMock()
        mock_response.result = "hello world"
        mock_response.exit_code = 0
        mock_sandbox.process.exec.return_value = mock_response
        backend._sandbox = mock_sandbox

        result = backend.run_command("echo hello world")
        assert isinstance(result, ExecutionResult)
        assert result.stdout == "hello world"
        assert result.exit_code == 0
        mock_sandbox.process.exec.assert_called_once_with(command="echo hello world")

    def test_run_command_not_started(self):
        backend = DaytonaBackend()
        try:
            backend.run_command("echo hi")
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    def test_run_code(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        mock_response = MagicMock()
        mock_response.result = "42"
        mock_response.exit_code = 0
        mock_sandbox.process.code_run.return_value = mock_response
        backend._sandbox = mock_sandbox

        result = backend.run_code("print(42)")
        assert isinstance(result, ExecutionResult)
        assert result.stdout == "42"
        assert result.exit_code == 0
        mock_sandbox.process.code_run.assert_called_once_with(code="print(42)")

    def test_write_file_str(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        backend._sandbox = mock_sandbox

        backend.write_file("/workspace/test.py", "print('hi')")
        mock_sandbox.fs.create_folder.assert_called_once_with(path="/workspace")
        mock_sandbox.fs.upload_file.assert_called_once_with(
            file=b"print('hi')", remote_path="/workspace/test.py"
        )

    def test_write_file_bytes(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        backend._sandbox = mock_sandbox

        content = b"\x89PNG\r\n"
        backend.write_file("/workspace/image.png", content)
        mock_sandbox.fs.upload_file.assert_called_once_with(
            file=content, remote_path="/workspace/image.png"
        )

    def test_write_file_not_started(self):
        backend = DaytonaBackend()
        try:
            backend.write_file("/test.py", "x")
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    def test_read_file(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        mock_sandbox.fs.download_file.return_value = b"file content"
        backend._sandbox = mock_sandbox

        result = backend.read_file("/workspace/test.py")
        assert result == "file content"
        mock_sandbox.fs.download_file.assert_called_once_with(remote_path="/workspace/test.py")

    def test_read_file_str_return(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        mock_sandbox.fs.download_file.return_value = "already a string"
        backend._sandbox = mock_sandbox

        result = backend.read_file("/workspace/test.py")
        assert result == "already a string"

    def test_close(self):
        backend = DaytonaBackend()
        mock_sandbox = MagicMock()
        backend._sandbox = mock_sandbox
        backend._daytona = MagicMock()

        backend.close()
        mock_sandbox.delete.assert_called_once()
        assert backend._sandbox is None
        assert backend._daytona is None

    def test_close_not_started(self):
        backend = DaytonaBackend()
        # Should not raise
        backend.close()
