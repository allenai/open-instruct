"""Unit tests for RL environments."""

import asyncio
import random
import unittest

from parameterized import parameterized

from open_instruct.environments.base import EnvCall, RLEnvironment, StepResult, TextRLEnvironment
from open_instruct.environments.examples import CounterEnv, GuessNumberEnv, WordleTextEnv


class TestEnvironmentReset(unittest.TestCase):
    """Test reset() contract for all environment subclasses."""

    @parameterized.expand(
        [
            ("counter", CounterEnv, {"target": 3}, {"target": "3"}),
            ("guess_number", GuessNumberEnv, {"min_val": 1, "max_val": 10}, {"number": "5"}),
            ("wordle", WordleTextEnv, {}, {"word": "CRANE"}),
        ]
    )
    def test_reset_returns_result_and_tools(self, _name, env_cls, kwargs, reset_kwargs):
        env = env_cls(**kwargs)
        result, tools = asyncio.run(env.reset(**reset_kwargs))
        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.result)
        self.assertIsInstance(tools, list)
        if isinstance(env, TextRLEnvironment):
            self.assertEqual(tools, [])
        else:
            self.assertGreater(len(tools), 0)


class TestEnvironments(unittest.TestCase):
    @parameterized.expand([
        (CounterEnv, {"target": 3}, {"target": "3"}),
        (GuessNumberEnv, {"min_val": 1, "max_val": 10}, {"number": "3"}),
    ])
    def test_20_random_steps(self, env_cls: type[RLEnvironment], kwargs: dict, reset_kwargs: dict):
        env = env_cls(**kwargs)
        result, tools = asyncio.run(env.reset(**reset_kwargs))
        self.assertIsInstance(result, StepResult)
        self.assertTrue(result.result)
        self.assertGreater(len(tools), 0)

        for _ in range(20):
            action = _random_env_call(tools)
            result = asyncio.run(env.step(action))
            self.assertIsInstance(result, StepResult)
            self.assertIsInstance(result.result, str)
            if result.done:
                result, tools = asyncio.run(env.reset(**reset_kwargs))


class TestWordleTextEnv(unittest.TestCase):
    def _make_env(self, word: str = "CRANE", **kwargs) -> WordleTextEnv:
        env = WordleTextEnv(**kwargs)
        asyncio.run(env.reset(word=word))
        return env

    @parameterized.expand(
        [
            ("correct_first", "<guess>CRANE</guess>", True, "You got it"),
            ("wrong", "<guess>HOUSE</guess>", False, "Feedback:"),
            ("case_insensitive", "<guess>crane</guess>", True, "You got it"),
            ("invalid_no_tags", "I guess CRANE", False, "<guess>"),
        ]
    )
    def test_guess(self, _name, text, expect_done, expect_in_result):
        env = self._make_env()
        r = asyncio.run(env.text_step(text))
        self.assertEqual(r.done, expect_done)
        self.assertIn(expect_in_result, r.result)

    def test_game_over_at_max_guesses(self):
        env = self._make_env(max_guesses=2)
        asyncio.run(env.text_step("<guess>HOUSE</guess>"))
        r = asyncio.run(env.text_step("<guess>BRAIN</guess>"))
        self.assertTrue(r.done)
        self.assertIn("Game over", r.result)
        self.assertIn("CRANE", r.result)

    def test_correct_reward_includes_turn_efficiency(self):
        env = self._make_env()
        r = asyncio.run(env.text_step("<guess>CRANE</guess>"))
        self.assertAlmostEqual(r.reward, 1.0 + 1.0 / 2)

        env2 = self._make_env()
        asyncio.run(env2.text_step("<guess>HOUSE</guess>"))
        asyncio.run(env2.text_step("<guess>BRAIN</guess>"))
        r2 = asyncio.run(env2.text_step("<guess>CRANE</guess>"))
        self.assertAlmostEqual(r2.reward, 1.0 + 1.0 / 4)

    def test_scoring_all_green(self):
        env = self._make_env()
        r = asyncio.run(env.text_step("<guess>CRANE</guess>"))
        self.assertIn("G G G G G", r.result)

    def test_scoring_yellow_and_miss(self):
        env = self._make_env()
        r = asyncio.run(env.text_step("<guess>NZZZZ</guess>"))
        self.assertIn("Feedback:", r.result)
        lines = r.result.split("Feedback:\n")[1].split("\n")
        self.assertEqual(lines[0], "N Z Z Z Z")
        self.assertIn("Y", lines[1])
        self.assertIn("X", lines[1])

    def test_step_delegates_to_text_step(self):
        env = self._make_env()
        r = asyncio.run(env.step(EnvCall(id="1", name="w", args={"text": "<guess>CRANE</guess>"})))
        self.assertTrue(r.done)

    def test_reset_no_tools(self):
        env = WordleTextEnv()
        _, tools = asyncio.run(env.reset(word="CRANE"))
        self.assertEqual(tools, [])

    def test_response_role(self):
        self.assertEqual(WordleTextEnv().get_response_role(), "user")

    def test_metrics(self):
        env = self._make_env()
        asyncio.run(env.text_step("no tags here"))
        asyncio.run(env.text_step("<guess>HOUSE</guess>"))
        asyncio.run(env.text_step("<guess>CRANE</guess>"))
        m = env.get_metrics()
        self.assertEqual(m["guesses"], 2.0)
        self.assertEqual(m["invalid_attempts"], 1.0)
        self.assertEqual(m["solved"], 1.0)
        self.assertGreater(m["total_greens"], 0)


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
