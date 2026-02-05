"""Unit tests for RL environments."""

import asyncio

from open_instruct.environments import ENV_REGISTRY, EnvironmentState, StepResult, ToolCall, get_env_class
from open_instruct.environments.base import RLEnvironment
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv
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
