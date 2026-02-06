"""Unit tests for RL environments."""

import asyncio
from unittest.mock import MagicMock, patch

from open_instruct.environments import ENV_REGISTRY, EnvironmentState, StepResult, ToolCall, get_env_class
from open_instruct.environments.backends import ExecutionResult
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
