"""Unit tests for RL environments."""

import asyncio
import random
import unittest

from parameterized import parameterized

from open_instruct.environments.base import EnvCall, RLEnvironment, StepResult
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv, WordleTextEnv


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


class TestWordleTextEnv(unittest.TestCase):

    def _make_env(self, word: str = "CRANE", **kwargs) -> WordleTextEnv:
        env = WordleTextEnv(**kwargs)
        asyncio.run(env.reset(task_id=word))
        return env

    @parameterized.expand([
        ("correct", "<guess>CRANE</guess>", True, 1.0, "You got it"),
        ("wrong", "<guess>HOUSE</guess>", False, 0.0, "remaining"),
        ("case_insensitive", "<guess>crane</guess>", True, 1.0, "You got it"),
        ("invalid_no_tags", "I guess CRANE", False, 0.0, "<guess>"),
    ])
    def test_guess(self, _name, text, expect_done, expect_reward, expect_in_result):
        env = self._make_env()
        r = asyncio.run(env.text_step(text))
        self.assertEqual(r.done, expect_done)
        self.assertEqual(r.reward, expect_reward)
        self.assertIn(expect_in_result, r.result)

    def test_game_over(self):
        env = self._make_env(max_guesses=2)
        asyncio.run(env.text_step("<guess>HOUSE</guess>"))
        r = asyncio.run(env.text_step("<guess>BRAIN</guess>"))
        self.assertTrue(r.done)
        self.assertIn("Game over", r.result)

    def test_scoring_all_green(self):
        env = self._make_env()
        r = asyncio.run(env.text_step("<guess>CRANE</guess>"))
        self.assertIn("GGGGG", r.result)

    def test_scoring_yellow_and_miss(self):
        env = self._make_env()
        r = asyncio.run(env.text_step("<guess>NXXXX</guess>"))
        self.assertIn("Feedback:", r.result)
        lines = r.result.split("Feedback:\n")[1].split("\n")
        self.assertEqual(lines[0], "NXXXX")
        self.assertIn("Y", lines[1])
        self.assertIn("_", lines[1])

    def test_step_delegates_to_text_step(self):
        env = self._make_env()
        r = asyncio.run(env.step(EnvCall(id="1", name="w", args={"text": "<guess>CRANE</guess>"})))
        self.assertTrue(r.done)

    def test_reset_no_tools(self):
        env = WordleTextEnv()
        _, tools = asyncio.run(env.reset(task_id="CRANE"))
        self.assertEqual(tools, [])

    def test_response_role(self):
        self.assertEqual(WordleTextEnv().get_response_role(), "user")

    def test_metrics(self):
        env = self._make_env()
        asyncio.run(env.text_step("<guess>HOUSE</guess>"))
        asyncio.run(env.text_step("<guess>CRANE</guess>"))
        self.assertEqual(env.get_metrics(), {"guesses": 2.0, "solved": 1.0})


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
