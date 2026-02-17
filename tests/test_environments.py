"""Unit tests for RL environments."""

import asyncio
import random

import pytest

from open_instruct.environments import ENV_REGISTRY, get_env_class
from open_instruct.environments.base import RLEnvironment, StepResult
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv
from open_instruct.data_types import ToolCall


def run_async(coro):
    """Run async function in sync test."""
    return asyncio.run(coro)


class TestRegistry:
    """Test environment registry."""

    def test_counter_registered(self):
        assert "counter" in ENV_REGISTRY

    def test_guess_number_registered(self):
        assert "guess_number" in ENV_REGISTRY

    def test_get_env_class_by_name(self):
        cls = get_env_class(env_name="counter")
        assert cls == CounterEnv

    def test_get_env_class_unknown_name(self):
        try:
            get_env_class(env_name="nonexistent")
            assert False, "Expected ValueError"
        except ValueError:
            pass



class TestCounterEnv:
    """Test CounterEnv."""

    def test_full_episode(self):
        async def _test():
            env = CounterEnv(target=3)
            result, tools = await env.reset()
            assert result.observation
            assert len(tools) == 3

            for _ in range(3):
                await env.step(ToolCall(id="", name="increment", args={}))

            step = await env.step(ToolCall(id="", name="submit", args={}))
            assert step.done
            assert step.reward == 1.0

        run_async(_test())

    def test_wrong_submit(self):
        async def _test():
            env = CounterEnv(target=5)
            await env.reset()
            step = await env.step(ToolCall(id="", name="submit", args={}))
            assert step.done
            assert step.reward == -0.5

        run_async(_test())

    def test_decrement(self):
        async def _test():
            env = CounterEnv(target=0)
            await env.reset()
            await env.step(ToolCall(id="", name="increment", args={}))
            await env.step(ToolCall(id="", name="decrement", args={}))
            step = await env.step(ToolCall(id="", name="submit", args={}))
            assert step.done
            assert step.reward == 1.0

        run_async(_test())

    def test_unknown_action(self):
        async def _test():
            env = CounterEnv()
            await env.reset()
            step = await env.step(ToolCall(id="", name="fly", args={}))
            assert not step.done
            assert step.reward == 0.0

        run_async(_test())

    def test_task_id_sets_target(self):
        async def _test():
            env = CounterEnv()
            result, _ = await env.reset(task_id="7")
            assert "7" in result.observation

        run_async(_test())

    def test_tool_definitions(self):
        tools = CounterEnv.get_tool_definitions()
        assert len(tools) == 3
        names = {t["function"]["name"] for t in tools}
        assert names == {"increment", "decrement", "submit"}

    def test_reset_returns_tool_definitions(self):
        async def _test():
            env = CounterEnv()
            _, tools = await env.reset()
            assert len(tools) == 3
            names = {t["function"]["name"] for t in tools}
            assert names == {"increment", "decrement", "submit"}

        run_async(_test())

    def test_metrics(self):
        async def _test():
            env = CounterEnv(target=2)
            await env.reset()
            await env.step(ToolCall(id="", name="increment", args={}))
            await env.step(ToolCall(id="", name="increment", args={}))
            await env.step(ToolCall(id="", name="submit", args={}))
            metrics = env.get_metrics()
            assert metrics["step_count"] == 3.0
            assert metrics["final_value"] == 2.0
            assert metrics["reached_target"] == 1.0

        run_async(_test())

    def test_state(self):
        async def _test():
            env = CounterEnv()
            await env.reset(task_id="5")
            await env.step(ToolCall(id="", name="increment", args={}))
            await env.step(ToolCall(id="", name="increment", args={}))
            s = env.state()
            assert s.episode_id == "5"
            assert s.step_count == 2
            assert s.info["current"] == 2
            assert s.info["target"] == 5

        run_async(_test())


class TestGuessNumberEnv:
    """Test GuessNumberEnv."""

    def test_correct_guess(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="5")
            result = await env.step(ToolCall(id="", name="guess", args={"number": 5}))
            assert result.done
            assert result.reward == 1.0

        run_async(_test())

    def test_too_low(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="50")
            result = await env.step(ToolCall(id="", name="guess", args={"number": 10}))
            assert not result.done
            assert "too low" in result.observation

        run_async(_test())

    def test_too_high(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="50")
            result = await env.step(ToolCall(id="", name="guess", args={"number": 90}))
            assert not result.done
            assert "too high" in result.observation

        run_async(_test())

    def test_closeness_reward(self):
        async def _test():
            env = GuessNumberEnv(min_val=1, max_val=100)
            await env.reset(task_id="50")
            close_result = await env.step(ToolCall(id="", name="guess", args={"number": 49}))
            await env.reset(task_id="50")
            far_result = await env.step(ToolCall(id="", name="guess", args={"number": 1}))
            assert close_result.reward > far_result.reward

        run_async(_test())

    def test_unknown_action(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="5")
            result = await env.step(ToolCall(id="", name="jump", args={}))
            assert not result.done
            assert result.reward == 0.0

        run_async(_test())

    def test_tool_definitions(self):
        tools = GuessNumberEnv.get_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "guess"

    def test_reset_returns_tool_definitions(self):
        async def _test():
            env = GuessNumberEnv()
            _, tools = await env.reset()
            assert len(tools) == 1
            assert tools[0]["function"]["name"] == "guess"

        run_async(_test())

    def test_metrics(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="5")
            await env.step(ToolCall(id="", name="guess", args={"number": 3}))
            await env.step(ToolCall(id="", name="guess", args={"number": 5}))
            metrics = env.get_metrics()
            assert metrics["guesses"] == 2.0

        run_async(_test())

    def test_state(self):
        async def _test():
            env = GuessNumberEnv()
            await env.reset(task_id="50")
            await env.step(ToolCall(id="", name="guess", args={"number": 30}))
            s = env.state()
            assert s.episode_id == "50"
            assert s.step_count == 1

        run_async(_test())


def _random_tool_call(tools: list[dict]) -> ToolCall:
    """Pick a random tool and fill required args with random values."""
    tool = random.choice(tools)
    func = tool["function"]
    args = {}
    for name, schema in func.get("parameters", {}).get("properties", {}).items():
        if schema.get("type") == "integer":
            args[name] = random.randint(-100, 100)
        elif schema.get("type") == "string":
            args[name] = "test"
        else:
            args[name] = None
    return ToolCall(id="", name=func["name"], args=args)


@pytest.mark.parametrize("env_cls", [CounterEnv, GuessNumberEnv])
def test_env_survives_20_random_steps(env_cls: type[RLEnvironment]):
    """Example envs should handle 20 random tool calls without crashing."""

    async def _test():
        env = env_cls()
        result, tools = await env.reset()
        assert isinstance(result, StepResult)
        assert len(tools) > 0

        for _ in range(20):
            tc = _random_tool_call(tools)
            step = await env.step(tc)
            assert isinstance(step, StepResult)
            assert isinstance(step.observation, str)
            if step.done:
                # Reset and keep going
                result, tools = await env.reset()

        # state() should work at any point
        s = env.state()
        assert s.step_count >= 0

    run_async(_test())
