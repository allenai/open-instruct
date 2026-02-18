"""Unit tests for RL environments."""

import asyncio
import random
import unittest

from parameterized import parameterized

from open_instruct.environments.base import EnvCall, RLEnvironment, StepResult
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv


class TestEnvCommon(unittest.TestCase):
    """Tests that apply to all example environments."""

    @parameterized.expand([(CounterEnv,), (GuessNumberEnv,)])
    def test_reset_returns_observation_and_tools(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        result, tools = asyncio.run(env.reset())
        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.observation)
        self.assertGreater(len(tools), 0)

    @parameterized.expand([(CounterEnv,), (GuessNumberEnv,)])
    def test_tool_definitions_match_reset(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        _, reset_tools = asyncio.run(env.reset())
        class_tools = env_cls.get_tool_definitions()
        self.assertEqual(reset_tools, class_tools)

    @parameterized.expand([(CounterEnv,), (GuessNumberEnv,)])
    def test_unknown_action_continues(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        asyncio.run(env.reset())
        step = asyncio.run(env.step(EnvCall(id="", name="nonexistent_tool", args={})))
        self.assertFalse(step.done)
        self.assertEqual(step.reward, 0.0)

    @parameterized.expand([(CounterEnv,), (GuessNumberEnv,)])
    def test_state_tracks_steps(self, env_cls: type[RLEnvironment]):
        env = env_cls()
        _, tools = asyncio.run(env.reset(task_id="5"))
        asyncio.run(env.step(EnvCall(id="", name=tools[0]["function"]["name"], args=_fill_args(tools[0]))))
        s = env.state()
        self.assertGreaterEqual(s.step_count, 1)

    @parameterized.expand([
        (CounterEnv, {"target": 3}),
        (GuessNumberEnv, {"min_val": 1, "max_val": 5}),
    ])
    def test_random_episode_completes(self, env_cls: type[RLEnvironment], kwargs: dict):
        """Random actions should eventually terminate the episode."""
        env = env_cls(**kwargs)
        result, tools = asyncio.run(env.reset(task_id="3"))
        for _ in range(500):
            action = _random_env_call(tools)
            result = asyncio.run(env.step(action))
            self.assertIsInstance(result, StepResult)
            self.assertIsInstance(result.observation, str)
            if result.done:
                break
        self.assertTrue(result.done, f"{env_cls.__name__} didn't terminate within 500 steps")
        self.assertTrue(env.state().done)

    @parameterized.expand([
        (CounterEnv, {"target": 2}),
        (GuessNumberEnv, {"min_val": 1, "max_val": 5}),
    ])
    def test_multiple_episodes(self, env_cls: type[RLEnvironment], kwargs: dict):
        """Can reset and run multiple episodes."""
        env = env_cls(**kwargs)
        for episode in range(3):
            result, tools = asyncio.run(env.reset(task_id="3"))
            self.assertFalse(result.done)
            for _ in range(500):
                action = _random_env_call(tools)
                result = asyncio.run(env.step(action))
                if result.done:
                    break
            self.assertTrue(result.done, f"Episode {episode} didn't terminate")


class TestCounterEnv(unittest.TestCase):
    def test_full_episode(self):
        env = CounterEnv(target=3)
        asyncio.run(env.reset())
        for _ in range(3):
            asyncio.run(env.step(EnvCall(id="", name="increment", args={})))
        step = asyncio.run(env.step(EnvCall(id="", name="submit", args={})))
        self.assertTrue(step.done)
        self.assertEqual(step.reward, 1.0)

    def test_wrong_submit(self):
        env = CounterEnv(target=5)
        asyncio.run(env.reset())
        step = asyncio.run(env.step(EnvCall(id="", name="submit", args={})))
        self.assertTrue(step.done)
        self.assertEqual(step.reward, -0.5)

    def test_decrement(self):
        env = CounterEnv(target=0)
        asyncio.run(env.reset())
        asyncio.run(env.step(EnvCall(id="", name="increment", args={})))
        asyncio.run(env.step(EnvCall(id="", name="decrement", args={})))
        step = asyncio.run(env.step(EnvCall(id="", name="submit", args={})))
        self.assertTrue(step.done)
        self.assertEqual(step.reward, 1.0)

    def test_task_id_sets_target(self):
        env = CounterEnv()
        result, _ = asyncio.run(env.reset(task_id="7"))
        self.assertIn("7", result.observation)

    def test_metrics(self):
        env = CounterEnv(target=2)
        asyncio.run(env.reset())
        asyncio.run(env.step(EnvCall(id="", name="increment", args={})))
        asyncio.run(env.step(EnvCall(id="", name="increment", args={})))
        asyncio.run(env.step(EnvCall(id="", name="submit", args={})))
        metrics = env.get_metrics()
        self.assertEqual(metrics["step_count"], 3.0)
        self.assertEqual(metrics["final_value"], 2.0)
        self.assertEqual(metrics["reached_target"], 1.0)


class TestGuessNumberEnv(unittest.TestCase):
    def test_correct_guess(self):
        env = GuessNumberEnv()
        asyncio.run(env.reset(task_id="5"))
        result = asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 5})))
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 1.0)

    def test_too_low(self):
        env = GuessNumberEnv()
        asyncio.run(env.reset(task_id="50"))
        result = asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 10})))
        self.assertFalse(result.done)
        self.assertIn("too low", result.observation)

    def test_too_high(self):
        env = GuessNumberEnv()
        asyncio.run(env.reset(task_id="50"))
        result = asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 90})))
        self.assertFalse(result.done)
        self.assertIn("too high", result.observation)

    def test_closeness_reward(self):
        env = GuessNumberEnv(min_val=1, max_val=100)
        asyncio.run(env.reset(task_id="50"))
        close_result = asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 49})))
        asyncio.run(env.reset(task_id="50"))
        far_result = asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 1})))
        self.assertGreater(close_result.reward, far_result.reward)

    def test_metrics(self):
        env = GuessNumberEnv()
        asyncio.run(env.reset(task_id="5"))
        asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 3})))
        asyncio.run(env.step(EnvCall(id="", name="guess", args={"number": 5})))
        metrics = env.get_metrics()
        self.assertEqual(metrics["guesses"], 2.0)


def _fill_args(tool_def: dict) -> dict:
    """Fill required args with dummy values for a tool definition."""
    args = {}
    for name, schema in tool_def["function"].get("parameters", {}).get("properties", {}).items():
        if schema.get("type") == "integer":
            args[name] = 1
        elif schema.get("type") == "string":
            args[name] = "test"
    return args


def _random_env_call(tools: list[dict]) -> EnvCall:
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
    return EnvCall(id="", name=func["name"], args=args)
