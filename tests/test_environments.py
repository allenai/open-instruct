"""Unit tests for RL environments."""

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

from datasets import Dataset

from open_instruct.dataset_transformation import ENV_CONFIG_KEY, rlvr_tokenize_v3
from open_instruct.environments import ENV_REGISTRY, EnvironmentState, StepResult, ToolCall, get_env_class
from open_instruct.environments.agent_task import AgentTaskEnv
from open_instruct.environments.backends import DaytonaBackend, ExecutionResult, create_backend
from open_instruct.environments.base import RLEnvironment
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv
from open_instruct.environments.openenv_client import OpenEnvClient, OpenEnvREPLClient, OpenEnvTextClient
from open_instruct.environments.sandbox_lm import SandboxLMEnv, _truncate_output
from open_instruct.ground_truth_utils import LastRewardAggregator, SumRewardAggregator
from open_instruct.tools.utils import EnvConfig, Tool


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


# ---------------------------------------------------------------------------
# EnvConfig merge logic tests
# ---------------------------------------------------------------------------
def _merge_env_config(base_env_config: dict | None, sample_env_config: dict | None) -> dict | None:
    """Replicate the merge logic from data_loader._put_prompt_request."""
    env_config = None
    if sample_env_config is not None:
        env_config = dict(base_env_config) if base_env_config else {}
        env_config.update(sample_env_config)
    return env_config


class TestEnvConfigMerge:
    """Test env_config merge logic (from data_loader._put_prompt_request)."""

    def test_merge_with_sample_and_base(self):
        """Sample env_config + base → merged with sample overriding base."""
        base = {"backend": "docker", "pool_size": 8, "timeout": 300}
        sample = {"task_id": "task_001", "env_name": "agent_task"}
        result = _merge_env_config(base, sample)
        assert result == {"backend": "docker", "pool_size": 8, "timeout": 300, "task_id": "task_001", "env_name": "agent_task"}

    def test_merge_without_sample(self):
        """No sample env_config + base → None (key new behavior)."""
        base = {"backend": "docker", "pool_size": 8, "env_name": "agent_task"}
        result = _merge_env_config(base, None)
        assert result is None

    def test_merge_no_base(self):
        """Sample env_config, no base → sample only."""
        sample = {"task_id": "42", "env_name": "counter"}
        result = _merge_env_config(None, sample)
        assert result == {"task_id": "42", "env_name": "counter"}

    def test_merge_sample_overrides_base(self):
        """Per-sample values override base values."""
        base = {"timeout": 60, "backend": "docker"}
        sample = {"task_id": "1", "env_name": "counter", "timeout": 120}
        result = _merge_env_config(base, sample)
        assert result["timeout"] == 120

    def test_merge_both_none(self):
        """Both None → None."""
        result = _merge_env_config(None, None)
        assert result is None


class TestEnvConfigEnabled:
    """Test EnvConfig.enabled property."""

    def test_enabled_with_env_name(self):
        config = EnvConfig(env_name="counter")
        assert config.enabled is True

    def test_enabled_with_env_class(self):
        config = EnvConfig(env_class="my.module.MyEnv")
        assert config.enabled is True

    def test_enabled_with_backend_only(self):
        """EnvConfig with only backend set → enabled (host-specific default)."""
        config = EnvConfig(env_backend="docker")
        assert config.enabled is True

    def test_enabled_with_task_data_dir_only(self):
        """EnvConfig with only task_data_dir set → enabled."""
        config = EnvConfig(env_task_data_dir="/data/tasks")
        assert config.enabled is True

    def test_enabled_with_base_url_only(self):
        """EnvConfig with only base_url set → enabled."""
        config = EnvConfig(env_base_url="http://localhost:8765")
        assert config.enabled is True

    def test_enabled_with_image_only(self):
        """EnvConfig with only image set → enabled."""
        config = EnvConfig(env_image="python:3.12-slim")
        assert config.enabled is True

    def test_not_enabled_default(self):
        """Default EnvConfig (no identity fields) → not enabled."""
        config = EnvConfig()
        assert config.enabled is False

    def test_not_enabled_only_operational_fields(self):
        """EnvConfig with only pool_size/max_steps/timeout → not enabled."""
        config = EnvConfig(env_pool_size=16, env_max_steps=20, env_timeout=120)
        assert config.enabled is False


class TestOpenEnvClientKwargs:
    """Test that OpenEnv clients accept **kwargs without raising."""

    def test_openenv_client_accepts_extra_kwargs(self):
        client = OpenEnvClient(
            base_url="http://localhost:8765",
            timeout=30,
            task_data_dir="/data/tasks",
            backend="docker",
        )
        assert client._base_url == "http://localhost:8765"
        assert client._timeout == 30

    def test_openenv_text_client_accepts_extra_kwargs(self):
        client = OpenEnvTextClient(
            base_url="http://localhost:8765",
            timeout=30,
            task_data_dir="/data/tasks",
            backend="docker",
        )
        assert client._base_url == "http://localhost:8765"

    def test_openenv_repl_client_accepts_extra_kwargs(self):
        client = OpenEnvREPLClient(
            base_url="http://localhost:8765",
            timeout=60,
            task_data_dir="/data/tasks",
            backend="docker",
        )
        assert client._base_url == "http://localhost:8765"


# ---------------------------------------------------------------------------
# discover_env_tool_definitions tests
# ---------------------------------------------------------------------------
class TestDiscoverEnvToolDefinitions:
    """Test discover_env_tool_definitions from grpo_fast."""

    def test_discovers_counter_env(self):
        """Discover counter env tools from a mock dataset."""
        from open_instruct.grpo_fast import discover_env_tool_definitions

        # Create a temporary dataset with env_config containing env_name
        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "Count to 3"}],
                [{"role": "user", "content": "Count to 5"}],
            ],
            "ground_truth": ["3", "5"],
            "dataset": ["passthrough", "passthrough"],
            "env_config": [
                {"task_id": "3", "env_name": "counter"},
                {"task_id": "5", "env_name": "counter"},
            ],
        })

        with patch("open_instruct.grpo_fast.datasets.load_dataset", return_value=ds):
            env_tool_map = discover_env_tool_definitions(
                dataset_mixer_list=["fake_dataset", "1.0"],
                dataset_mixer_list_splits=["train"],
                env_config=EnvConfig(),
            )

        assert "counter" in env_tool_map
        tool_names = {t["function"]["name"] for t in env_tool_map["counter"]}
        assert "increment" in tool_names
        assert "submit" in tool_names

    def test_discovers_multiple_envs(self):
        """Discover tools from a dataset mixing two env types."""
        from open_instruct.grpo_fast import discover_env_tool_definitions

        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "Count to 3"}],
                [{"role": "user", "content": "Guess 42"}],
            ],
            "ground_truth": ["3", "42"],
            "dataset": ["passthrough", "passthrough"],
            "env_config": [
                {"task_id": "3", "env_name": "counter"},
                {"task_id": "42", "env_name": "guess_number"},
            ],
        })

        with patch("open_instruct.grpo_fast.datasets.load_dataset", return_value=ds):
            env_tool_map = discover_env_tool_definitions(
                dataset_mixer_list=["fake_dataset", "1.0"],
                dataset_mixer_list_splits=["train"],
                env_config=EnvConfig(),
            )

        assert "counter" in env_tool_map
        assert "guess_number" in env_tool_map
        counter_names = {t["function"]["name"] for t in env_tool_map["counter"]}
        guess_names = {t["function"]["name"] for t in env_tool_map["guess_number"]}
        assert "increment" in counter_names
        assert "guess" in guess_names

    def test_no_env_config_column(self):
        """Dataset without env_config column returns empty map."""
        from open_instruct.grpo_fast import discover_env_tool_definitions

        ds = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Hello"}]],
            "ground_truth": ["hi"],
            "dataset": ["math"],
        })

        with patch("open_instruct.grpo_fast.datasets.load_dataset", return_value=ds):
            env_tool_map = discover_env_tool_definitions(
                dataset_mixer_list=["fake_dataset", "1.0"],
                dataset_mixer_list_splits=["train"],
                env_config=EnvConfig(),
            )

        assert env_tool_map == {}

    def test_cli_env_name_fallback(self):
        """CLI --env_name should be included even without dataset env_config."""
        from open_instruct.grpo_fast import discover_env_tool_definitions

        ds = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Hello"}]],
            "ground_truth": ["hi"],
            "dataset": ["math"],
        })

        with patch("open_instruct.grpo_fast.datasets.load_dataset", return_value=ds):
            env_tool_map = discover_env_tool_definitions(
                dataset_mixer_list=["fake_dataset", "1.0"],
                dataset_mixer_list_splits=["train"],
                env_config=EnvConfig(env_name="counter"),
            )

        assert "counter" in env_tool_map


# ---------------------------------------------------------------------------
# Per-sample env tool injection in rlvr_tokenize_v3 tests
# ---------------------------------------------------------------------------
class TestPerSampleEnvToolInjection:
    """Test that rlvr_tokenize_v3 injects env-specific tools per sample."""

    def _make_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        return tokenizer

    def _counter_tool_map(self):
        return {
            "counter": [
                {"type": "function", "function": {"name": "increment", "parameters": {}}},
                {"type": "function", "function": {"name": "submit", "parameters": {}}},
            ],
        }

    def _guess_tool_map(self):
        return {
            "guess_number": [
                {"type": "function", "function": {"name": "guess", "parameters": {}}},
            ],
        }

    def test_env_tools_injected_for_matching_sample(self):
        """Sample with env_config.env_name=counter should get counter tools."""
        tokenizer = self._make_tokenizer()
        env_tool_map = self._counter_tool_map()

        row = {
            "messages": [{"role": "user", "content": "Count to 3"}],
            "ground_truth": "3",
            "dataset": "passthrough",
            "env_config": {"task_id": "3", "env_name": "counter"},
        }

        rlvr_tokenize_v3(row, tokenizer, env_tool_map=env_tool_map)

        # Check that apply_chat_template was called with counter tools
        call_kwargs = tokenizer.apply_chat_template.call_args
        tools_passed = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_passed is not None
        tool_names = {t["function"]["name"] for t in tools_passed}
        assert "increment" in tool_names
        assert "submit" in tool_names

    def test_no_env_tools_for_non_env_sample(self):
        """Sample without env_config should not get env tools."""
        tokenizer = self._make_tokenizer()
        env_tool_map = self._counter_tool_map()

        row = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "ground_truth": "4",
            "dataset": "math",
        }

        rlvr_tokenize_v3(row, tokenizer, env_tool_map=env_tool_map)

        call_kwargs = tokenizer.apply_chat_template.call_args
        tools_passed = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_passed is None

    def test_env_tools_combined_with_regular_tools(self):
        """Env tools should merge with regular tool_definitions."""
        tokenizer = self._make_tokenizer()
        env_tool_map = self._counter_tool_map()
        regular_tools = [
            {"type": "function", "function": {"name": "python", "parameters": {}}},
        ]

        row = {
            "messages": [{"role": "user", "content": "Count to 3"}],
            "ground_truth": "3",
            "dataset": "passthrough",
            "env_config": {"task_id": "3", "env_name": "counter"},
        }

        rlvr_tokenize_v3(row, tokenizer, tool_definitions=regular_tools, env_tool_map=env_tool_map)

        call_kwargs = tokenizer.apply_chat_template.call_args
        tools_passed = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_passed is not None
        tool_names = {t["function"]["name"] for t in tools_passed}
        # Should have both regular and env tools
        assert "python" in tool_names
        assert "increment" in tool_names
        assert "submit" in tool_names

    def test_different_envs_get_different_tools(self):
        """Counter and guess_number samples get their respective tools."""
        tokenizer = self._make_tokenizer()
        env_tool_map = {**self._counter_tool_map(), **self._guess_tool_map()}

        counter_row = {
            "messages": [{"role": "user", "content": "Count to 3"}],
            "ground_truth": "3",
            "dataset": "passthrough",
            "env_config": {"task_id": "3", "env_name": "counter"},
        }

        guess_row = {
            "messages": [{"role": "user", "content": "Guess 42"}],
            "ground_truth": "42",
            "dataset": "passthrough",
            "env_config": {"task_id": "42", "env_name": "guess_number"},
        }

        rlvr_tokenize_v3(counter_row, tokenizer, env_tool_map=env_tool_map)
        counter_call = tokenizer.apply_chat_template.call_args
        counter_tools = counter_call.kwargs.get("tools") or counter_call[1].get("tools")
        counter_names = {t["function"]["name"] for t in counter_tools}

        tokenizer.reset_mock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]

        rlvr_tokenize_v3(guess_row, tokenizer, env_tool_map=env_tool_map)
        guess_call = tokenizer.apply_chat_template.call_args
        guess_tools = guess_call.kwargs.get("tools") or guess_call[1].get("tools")
        guess_names = {t["function"]["name"] for t in guess_tools}

        assert "increment" in counter_names
        assert "guess" not in counter_names
        assert "guess" in guess_names
        assert "increment" not in guess_names

    def test_no_duplicate_tools_when_env_tool_already_in_definitions(self):
        """Env tools already present in tool_definitions should not be duplicated."""
        tokenizer = self._make_tokenizer()
        env_tool_map = self._counter_tool_map()
        # Include increment in the regular tool_definitions too
        regular_tools = [
            {"type": "function", "function": {"name": "increment", "parameters": {}}},
        ]

        row = {
            "messages": [{"role": "user", "content": "Count to 3"}],
            "ground_truth": "3",
            "dataset": "passthrough",
            "env_config": {"task_id": "3", "env_name": "counter"},
        }

        rlvr_tokenize_v3(row, tokenizer, tool_definitions=regular_tools, env_tool_map=env_tool_map)

        call_kwargs = tokenizer.apply_chat_template.call_args
        tools_passed = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_passed is not None
        # Count how many times "increment" appears
        increment_count = sum(1 for t in tools_passed if t["function"]["name"] == "increment")
        assert increment_count == 1
