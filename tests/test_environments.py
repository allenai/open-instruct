"""Unit tests for RL environments."""

import asyncio

import pytest

from open_instruct.environments import (
    ENV_REGISTRY,
    EnvironmentState,
    ResetResult,
    StepResult,
    ToolCall,
    get_env_class,
)
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv


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
            assert isinstance(result, ResetResult)
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


class TestVerifiers:
    """Test environment verifiers."""

    def test_last_reward_verifier(self):
        from open_instruct.ground_truth_utils import LastRewardEnvVerifier

        verifier = LastRewardEnvVerifier()
        state = EnvironmentState(rewards=[0.1, 0.5, 1.0])
        result = verifier([], "", state)
        assert result.score == 1.0

    def test_sum_reward_verifier(self):
        from open_instruct.ground_truth_utils import SumRewardEnvVerifier

        verifier = SumRewardEnvVerifier()
        state = EnvironmentState(rewards=[1.0, 2.0, 3.0])
        result = verifier([], "", state)
        assert result.score == 6.0
