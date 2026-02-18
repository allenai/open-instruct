"""Unit tests for RL environments."""

import asyncio
import random
import unittest

from parameterized import parameterized

from open_instruct.environments.base import EnvCall, RLEnvironment, StepResult
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv


class TestEnvironments(unittest.TestCase):

    @parameterized.expand([
        (CounterEnv, {"target": 3}),
        (GuessNumberEnv, {"min_val": 1, "max_val": 10}),
    ])
    def test_20_random_steps(self, env_cls: type[RLEnvironment], kwargs: dict):
        env = env_cls(**kwargs)
        result, tools = asyncio.run(env.reset(task_id="3"))
        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.result)
        self.assertGreater(len(tools), 0)

        for _ in range(20):
            action = _random_env_call(tools)
            result = asyncio.run(env.step(action))
            self.assertIsInstance(result, StepResult)
            self.assertIsInstance(result.result, str)
            if result.done:
                result, tools = asyncio.run(env.reset(task_id="3"))


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
    return EnvCall(id="", name=func["name"], args=args)
