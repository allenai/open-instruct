"""
Unit tests for RL environments.

Run with: pytest tests/test_environments.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_instruct.environments import (
    ENV_REGISTRY,
    EnvironmentState,
    ResetResult,
    StepResult,
    ToolCall,
    get_env_class,
    register_env,
)
from open_instruct.environments.base import RLEnvironment
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_basic_tool_call(self):
        tc = ToolCall(name="test", args={"arg1": "value1"})
        assert tc.name == "test"
        assert tc.args == {"arg1": "value1"}
        assert tc.id is None

    def test_tool_call_with_id(self):
        tc = ToolCall(name="test", args={}, id="call_123")
        assert tc.id == "call_123"


class TestResetResult:
    """Tests for ResetResult dataclass."""

    def test_basic_reset_result(self):
        result = ResetResult(
            observation="Hello",
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        assert result.observation == "Hello"
        assert len(result.tools) == 1
        assert result.info == {}

    def test_reset_result_with_info(self):
        result = ResetResult(
            observation="Hello",
            tools=[],
            info={"task_id": "123"},
        )
        assert result.info["task_id"] == "123"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_basic_step_result(self):
        result = StepResult(observation="Done", reward=1.0, done=True)
        assert result.observation == "Done"
        assert result.reward == 1.0
        assert result.done is True
        assert result.info == {}

    def test_step_result_with_info(self):
        result = StepResult(
            observation="Step",
            reward=0.5,
            done=False,
            info={"step": 1},
        )
        assert result.info["step"] == 1


class TestEnvironmentState:
    """Tests for EnvironmentState dataclass."""

    def test_empty_state(self):
        state = EnvironmentState()
        assert state.rewards == []
        assert state.step_count == 0
        assert state.done is False
        assert state.final_reward == 0.0
        assert state.total_reward == 0.0

    def test_state_with_rewards(self):
        state = EnvironmentState(rewards=[0.1, 0.2, 0.5])
        assert state.final_reward == 0.5
        assert state.total_reward == 0.8

    def test_state_single_reward(self):
        state = EnvironmentState(rewards=[1.0])
        assert state.final_reward == 1.0
        assert state.total_reward == 1.0


class TestEnvRegistry:
    """Tests for environment registry."""

    def test_builtin_envs_registered(self):
        """Check that built-in environments are registered."""
        assert "counter" in ENV_REGISTRY
        assert "guess_number" in ENV_REGISTRY
        assert "sandbox" in ENV_REGISTRY
        assert "openenv" in ENV_REGISTRY

    def test_get_env_class_by_name(self):
        """Test getting environment class by registered name."""
        cls = get_env_class(env_name="counter")
        assert cls == CounterEnv

    def test_get_env_class_not_found(self):
        """Test error when environment not found."""
        with pytest.raises(ValueError, match="not found in registry"):
            get_env_class(env_name="nonexistent")

    def test_get_env_class_requires_arg(self):
        """Test error when no argument provided."""
        with pytest.raises(ValueError, match="Must provide"):
            get_env_class()


class TestCounterEnv:
    """Tests for CounterEnv example environment."""

    @pytest.fixture
    def env(self):
        """Create a CounterEnv instance."""
        return CounterEnv(target=3)

    def test_reset(self, env):
        """Test environment reset."""
        result = run_async(env.reset())

        assert isinstance(result, ResetResult)
        assert "0" in result.observation  # Counter starts at 0
        assert "3" in result.observation  # Target is mentioned
        assert len(result.tools) == 3  # increment, decrement, submit
        assert result.info["target"] == 3

    def test_reset_with_task_id(self, env):
        """Test reset with custom target via task_id."""
        result = run_async(env.reset(task_id="10"))
        assert result.info["target"] == 10

    def test_increment(self, env):
        """Test increment action."""
        run_async(env.reset())
        result = run_async(env.step(ToolCall(name="increment", args={})))

        assert isinstance(result, StepResult)
        assert "1" in result.observation
        assert result.reward == -0.1  # Step penalty
        assert result.done is False

    def test_decrement(self, env):
        """Test decrement action."""
        run_async(env.reset())
        result = run_async(env.step(ToolCall(name="decrement", args={})))

        assert "-1" in result.observation
        assert result.done is False

    def test_reach_target(self, env):
        """Test reaching the target."""
        run_async(env.reset())

        # Increment 3 times to reach target
        for _ in range(3):
            run_async(env.step(ToolCall(name="increment", args={})))

        # Submit
        result = run_async(env.step(ToolCall(name="submit", args={})))

        assert result.reward == 1.0
        assert result.done is True
        assert "Success" in result.observation

    def test_wrong_submit(self, env):
        """Test submitting wrong answer."""
        run_async(env.reset())
        result = run_async(env.step(ToolCall(name="submit", args={})))

        assert result.reward == -0.5
        assert result.done is True
        assert "Wrong" in result.observation

    def test_metrics(self, env):
        """Test get_metrics."""
        run_async(env.reset())
        run_async(env.step(ToolCall(name="increment", args={})))
        run_async(env.step(ToolCall(name="increment", args={})))

        metrics = env.get_metrics()
        assert metrics["step_count"] == 2
        assert metrics["final_value"] == 2
        assert metrics["reached_target"] == 0.0


class TestGuessNumberEnv:
    """Tests for GuessNumberEnv example environment."""

    @pytest.fixture
    def env(self):
        """Create a GuessNumberEnv instance."""
        return GuessNumberEnv(min_val=1, max_val=10)

    def test_reset(self, env):
        """Test environment reset."""
        result = run_async(env.reset())

        assert isinstance(result, ResetResult)
        assert "1" in result.observation and "10" in result.observation
        assert len(result.tools) == 1  # guess tool
        assert result.tools[0]["function"]["name"] == "guess"

    def test_reset_with_known_number(self, env):
        """Test reset with known secret via task_id."""
        run_async(env.reset(task_id="5"))
        # The secret should be 5
        assert env._secret == 5

    def test_guess_too_low(self, env):
        """Test guessing too low."""
        run_async(env.reset(task_id="5"))
        result = run_async(env.step(ToolCall(name="guess", args={"number": 2})))

        assert "too low" in result.observation.lower()
        assert result.reward == 0.0
        assert result.done is False

    def test_guess_too_high(self, env):
        """Test guessing too high."""
        run_async(env.reset(task_id="5"))
        result = run_async(env.step(ToolCall(name="guess", args={"number": 8})))

        assert "too high" in result.observation.lower()
        assert result.reward == 0.0
        assert result.done is False

    def test_guess_correct(self, env):
        """Test correct guess."""
        run_async(env.reset(task_id="5"))
        result = run_async(env.step(ToolCall(name="guess", args={"number": 5})))

        assert "correct" in result.observation.lower()
        assert result.reward == 1.0
        assert result.done is True

    def test_invalid_guess(self, env):
        """Test invalid guess type."""
        run_async(env.reset())
        result = run_async(env.step(ToolCall(name="guess", args={"number": "not a number"})))

        assert "invalid" in result.observation.lower()
        assert result.reward == -0.1
        assert result.done is False


class TestEnvironmentVerifiers:
    """Tests for environment verifiers."""

    def test_last_reward_verifier_with_state(self):
        """Test LastRewardEnvVerifier with EnvironmentState."""
        from open_instruct.ground_truth_utils import LastRewardEnvVerifier

        verifier = LastRewardEnvVerifier()
        state = EnvironmentState(rewards=[0.1, 0.2, 0.8])

        result = verifier([], "", state)
        assert result.score == 0.8

    def test_last_reward_verifier_empty(self):
        """Test LastRewardEnvVerifier with empty rewards."""
        from open_instruct.ground_truth_utils import LastRewardEnvVerifier

        verifier = LastRewardEnvVerifier()
        state = EnvironmentState(rewards=[])

        result = verifier([], "", state)
        assert result.score == 0.0

    def test_last_reward_verifier_with_dict(self):
        """Test LastRewardEnvVerifier with dict representation."""
        from open_instruct.ground_truth_utils import LastRewardEnvVerifier

        verifier = LastRewardEnvVerifier()
        state_dict = {"rewards": [0.1, 0.5, 1.0]}

        result = verifier([], "", state_dict)
        assert result.score == 1.0

    def test_sum_reward_verifier_with_state(self):
        """Test SumRewardEnvVerifier with EnvironmentState."""
        from open_instruct.ground_truth_utils import SumRewardEnvVerifier

        verifier = SumRewardEnvVerifier()
        state = EnvironmentState(rewards=[0.1, 0.2, 0.3])

        result = verifier([], "", state)
        assert abs(result.score - 0.6) < 0.001

    def test_sum_reward_verifier_empty(self):
        """Test SumRewardEnvVerifier with empty rewards."""
        from open_instruct.ground_truth_utils import SumRewardEnvVerifier

        verifier = SumRewardEnvVerifier()
        state = EnvironmentState(rewards=[])

        result = verifier([], "", state)
        assert result.score == 0.0

    def test_sum_reward_verifier_with_dict(self):
        """Test SumRewardEnvVerifier with dict representation."""
        from open_instruct.ground_truth_utils import SumRewardEnvVerifier

        verifier = SumRewardEnvVerifier()
        state_dict = {"rewards": [1.0, 2.0, 3.0]}

        result = verifier([], "", state_dict)
        assert result.score == 6.0


class TestSandboxBackend:
    """Tests for sandbox backends (mocked)."""

    def test_create_e2b_backend(self):
        """Test creating E2B backend."""
        from open_instruct.environments.backends import E2BBackend

        backend = E2BBackend(template="base", timeout=300)
        assert backend._template == "base"
        assert backend._timeout == 300
        assert backend._sandbox is None

    def test_create_docker_backend(self):
        """Test creating Docker backend."""
        from open_instruct.environments.backends import DockerBackend

        backend = DockerBackend(image="test:latest")
        assert backend._image == "test:latest"
        assert backend._runtime is None

    def test_create_backend_factory(self):
        """Test create_backend factory function."""
        from open_instruct.environments.backends import E2BBackend, DockerBackend, create_backend

        e2b = create_backend("e2b", template="custom")
        assert isinstance(e2b, E2BBackend)
        assert e2b._template == "custom"

        docker = create_backend("docker", image="custom:tag")
        assert isinstance(docker, DockerBackend)
        assert docker._image == "custom:tag"

    def test_create_backend_invalid(self):
        """Test create_backend with invalid type."""
        from open_instruct.environments.backends import create_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("invalid")


class TestEnvironmentTool:
    """Tests for EnvironmentTool configuration."""

    def test_environment_tool_config(self):
        """Test EnvironmentToolConfig creation."""
        from open_instruct.tools.tools import EnvironmentToolConfig

        config = EnvironmentToolConfig(
            env_name="counter",
            pool_size=32,
            response_role="tool",
        )

        assert config.env_name == "counter"
        assert config.pool_size == 32
        assert config.response_role == "tool"

    def test_environment_tool_in_registry(self):
        """Test that environment tool is in TOOL_REGISTRY."""
        from open_instruct.tools.tools import TOOL_REGISTRY

        assert "environment" in TOOL_REGISTRY

    def test_environment_tool_creation(self):
        """Test creating EnvironmentTool instance."""
        from open_instruct.tools.tools import EnvironmentTool

        tool = EnvironmentTool(
            call_name="test_env",
            env_name="counter",
            pool_size=4,
        )

        assert tool.call_name == "test_env"
        assert tool.env_name == "counter"
        assert tool.pool_size == 4
