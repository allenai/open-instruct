"""Unit tests for RL environments."""

import random

import pytest

from open_instruct.environments import ENV_REGISTRY, get_env_class
from open_instruct.environments.base import RLEnvironment, StepResult
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv
from open_instruct.environments.tools.utils import ToolCall


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
        with pytest.raises(ValueError):
            get_env_class(env_name="nonexistent")


@pytest.mark.parametrize("env_cls", [CounterEnv, GuessNumberEnv])
class TestEnvCommon:
    """Tests that apply to all example environments."""

    @pytest.mark.asyncio
    async def test_reset_returns_observation_and_tools(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        result, tools = await env.reset()
        assert isinstance(result, StepResult)
        assert result.observation
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_tool_definitions_match_reset(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        _, reset_tools = await env.reset()
        class_tools = env_cls.get_tool_definitions()
        assert reset_tools == class_tools

    @pytest.mark.asyncio
    async def test_unknown_action_continues(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        await env.reset()
        step = await env.step(ToolCall(id="", name="nonexistent_tool", args={}))
        assert not step.done
        assert step.reward == 0.0

    @pytest.mark.asyncio
    async def test_state_tracks_steps(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        _, tools = await env.reset(task_id="5")
        await env.step(ToolCall(id="", name=tools[0]["function"]["name"], args=_fill_args(tools[0])))
        s = env.state()
        assert s.step_count >= 1

    @pytest.mark.asyncio
    async def test_survives_20_random_steps(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        result, tools = await env.reset()
        for _ in range(20):
            tc = _random_tool_call(tools)
            step = await env.step(tc)
            assert isinstance(step, StepResult)
            assert isinstance(step.observation, str)
            if step.done:
                result, tools = await env.reset()
        s = env.state()
        assert s.step_count >= 0


class TestCounterEnv:
    @pytest.mark.asyncio
    async def test_full_episode(self):
        env = CounterEnv(target=3)
        await env.reset()
        for _ in range(3):
            await env.step(ToolCall(id="", name="increment", args={}))
        step = await env.step(ToolCall(id="", name="submit", args={}))
        assert step.done
        assert step.reward == 1.0

    @pytest.mark.asyncio
    async def test_wrong_submit(self):
        env = CounterEnv(target=5)
        await env.reset()
        step = await env.step(ToolCall(id="", name="submit", args={}))
        assert step.done
        assert step.reward == -0.5

    @pytest.mark.asyncio
    async def test_decrement(self):
        env = CounterEnv(target=0)
        await env.reset()
        await env.step(ToolCall(id="", name="increment", args={}))
        await env.step(ToolCall(id="", name="decrement", args={}))
        step = await env.step(ToolCall(id="", name="submit", args={}))
        assert step.done
        assert step.reward == 1.0

    @pytest.mark.asyncio
    async def test_task_id_sets_target(self):
        env = CounterEnv()
        result, _ = await env.reset(task_id="7")
        assert "7" in result.observation

    @pytest.mark.asyncio
    async def test_metrics(self):
        env = CounterEnv(target=2)
        await env.reset()
        await env.step(ToolCall(id="", name="increment", args={}))
        await env.step(ToolCall(id="", name="increment", args={}))
        await env.step(ToolCall(id="", name="submit", args={}))
        metrics = env.get_metrics()
        assert metrics["step_count"] == 3.0
        assert metrics["final_value"] == 2.0
        assert metrics["reached_target"] == 1.0


class TestGuessNumberEnv:
    @pytest.mark.asyncio
    async def test_correct_guess(self):
        env = GuessNumberEnv()
        await env.reset(task_id="5")
        result = await env.step(ToolCall(id="", name="guess", args={"number": 5}))
        assert result.done
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_too_low(self):
        env = GuessNumberEnv()
        await env.reset(task_id="50")
        result = await env.step(ToolCall(id="", name="guess", args={"number": 10}))
        assert not result.done
        assert "too low" in result.observation

    @pytest.mark.asyncio
    async def test_too_high(self):
        env = GuessNumberEnv()
        await env.reset(task_id="50")
        result = await env.step(ToolCall(id="", name="guess", args={"number": 90}))
        assert not result.done
        assert "too high" in result.observation

    @pytest.mark.asyncio
    async def test_closeness_reward(self):
        env = GuessNumberEnv(min_val=1, max_val=100)
        await env.reset(task_id="50")
        close_result = await env.step(ToolCall(id="", name="guess", args={"number": 49}))
        await env.reset(task_id="50")
        far_result = await env.step(ToolCall(id="", name="guess", args={"number": 1}))
        assert close_result.reward > far_result.reward

    @pytest.mark.asyncio
    async def test_metrics(self):
        env = GuessNumberEnv()
        await env.reset(task_id="5")
        await env.step(ToolCall(id="", name="guess", args={"number": 3}))
        await env.step(ToolCall(id="", name="guess", args={"number": 5}))
        metrics = env.get_metrics()
        assert metrics["guesses"] == 2.0


def _fill_args(tool_def: dict) -> dict:
    """Fill required args with dummy values for a tool definition."""
    args = {}
    for name, schema in tool_def["function"].get("parameters", {}).get("properties", {}).items():
        if schema.get("type") == "integer":
            args[name] = 1
        elif schema.get("type") == "string":
            args[name] = "test"
    return args


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
