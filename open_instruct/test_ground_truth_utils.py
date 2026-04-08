#!/usr/bin/env python3
"""
Test script for verifier functionality in Python
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from parameterized import parameterized

from open_instruct.data_types import RequestInfo
from open_instruct.ground_truth_utils import (
    BallsimVerifier,
    BallsimVerifierConfig,
    F1Verifier,
    GSM8KVerifier,
    LLMJudgeFallbackVerifier,
    LMJudgeVerifier,
    ManufactoriaVerifier,
    ManufactoriaVerifierConfig,
    PuzzleMatcherVerifier,
    RewardConfig,
    VerificationResult,
    apply_verifiable_reward,
    build_all_verifiers,
)
from open_instruct.judge_utils import extract_score_compass_verifier


class TestPuzzleMatcherVerifier(unittest.TestCase):
    """Test suite for PuzzleMatcherVerifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = PuzzleMatcherVerifier()

    @parameterized.expand(
        [
            ("simple_match", "The answer is 42", "answer is 42", 1.0),
            ("with_thinking_tags", "<think>Let me solve this</think>Paris", "paris", 1.0),
            ("with_answer_tags", "<answer>New York City!</answer>", "new york city", 1.0),
            ("should_fail", "Wrong answer", "correct answer", 0.0),
            (
                "complex_example",
                "<think>This is about geography</think><answer>The capital of France is Paris.</answer>",
                "capital of france is paris",
                1.0,
            ),
        ]
    )
    def test_basic_scenarios(self, name, prediction, label, expected_score):
        """Test basic puzzle matcher scenarios from quick_test"""
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)

    @parameterized.expand(
        [
            # Basic matching tests
            ("exact_match_numbers", "42", "42", 1.0),
            ("exact_match_text", "hello world", "hello world", 1.0),
            ("case_insensitive", "Hello World", "hello world", 1.0),
            # Tests with thinking tags
            ("thinking_tags_match", "<think>Let me think about this...</think>42", "42", 1.0),
            ("thinking_tags_text_match", "<think>This is complex</think>hello world", "hello world", 1.0),
            ("thinking_tags_no_match", "<think>Analysis...</think>Wrong Answer", "42", 0.0),
            # Tests with answer tags
            ("answer_tags_match", "<answer>42</answer>", "42", 1.0),
            ("answer_tags_text_match", "<answer>hello world</answer>", "hello world", 1.0),
            ("answer_tags_no_match", "<answer>wrong</answer>", "42", 0.0),
            # Combined tags tests
            ("both_tags_match", "<think>Thinking...</think><answer>42</answer>", "42", 1.0),
            (
                "both_tags_text_match",
                "<think>Let me solve this step by step</think><answer>hello world</answer>",
                "hello world",
                1.0,
            ),
            # Punctuation and articles tests
            ("remove_articles_punctuation", "The answer is 42!", "answer is 42", 1.0),
            ("remove_article_a", "A simple test.", "simple test", 1.0),
            ("remove_punctuation", "Hello, world!", "hello world", 1.0),
            # Whitespace tests
            ("normalize_whitespace", "  hello   world  ", "hello world", 1.0),
            ("replace_tabs_newlines", "hello\tworld\n", "hello world", 1.0),
            # Non-matching tests
            ("numbers_no_match", "42", "43", 0.0),
            ("text_no_match", "hello", "world", 0.0),
            ("empty_vs_nonempty", "", "42", 0.0),
            # English examples
            ("capital_city", "<answer>London</answer>", "london", 1.0),
            ("animal_with_article", "<think>Animal question</think>The elephant", "elephant", 1.0),
            ("scientist_name", "<answer>Albert Einstein</answer>", "albert einstein", 1.0),
            ("literature_reference", "Romeo and Juliet by Shakespeare", "romeo and juliet by shakespeare", 1.0),
            ("country_name", "<answer>United States of America</answer>", "united states of america", 1.0),
        ]
    )
    def test_puzzle_matcher_scenarios(self, name, prediction, label, expected_score):
        """Test various puzzle matcher scenarios"""
        result = self.verifier([], prediction, label)
        self.assertEqual(
            result.score, expected_score, f"Failed for {name}: prediction='{prediction}', label='{label}'"
        )


class TestF1Verifier(unittest.TestCase):
    """Test suite for F1Verifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = F1Verifier()

    @parameterized.expand(
        [
            # Basic F1 tests with single string label
            ("exact_match", "hello world", "hello world", 1.0),
            ("partial_match", "hello world", "hello", 2 / 3),  # precision=0.5, recall=1.0, f1=2/3
            ("no_match", "hello world", "goodbye", 0.0),
            # Thinking section removal
            ("with_thinking", "<think>Let me think...</think>hello world", "hello world", 1.0),
            ("with_thinking_partial", "<think>Analysis</think>hello world", "hello", 2 / 3),
            # Answer tag removal
            ("with_answer_tags", "<answer>hello world</answer>", "hello world", 1.0),
            # Combined tags
            ("both_tags", "<think>Thinking...</think><answer>hello world</answer>", "hello world", 1.0),
        ]
    )
    def test_single_label(self, name, prediction, label, expected_score):
        """Test F1 verifier with single string label"""
        result = self.verifier([], prediction, label)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', label='{label}'",
        )

    @parameterized.expand(
        [
            # List of labels - should return max F1
            ("first_matches_best", "hello world", ["hello world", "goodbye"], 1.0),
            ("second_matches_best", "hello world", ["goodbye", "hello world"], 1.0),
            ("partial_matches", "hello world", ["hello", "world"], 2 / 3),  # both have same F1
            ("none_match_well", "hello world", ["foo", "bar", "baz"], 0.0),
            # Single element list should behave same as string
            ("single_element_list", "hello world", ["hello world"], 1.0),
            # With thinking section
            ("list_with_thinking", "<think>hmm</think>hello world", ["goodbye", "hello world"], 1.0),
        ]
    )
    def test_list_labels(self, name, prediction, labels, expected_score):
        """Test F1 verifier with list of labels (should return max)"""
        result = self.verifier([], prediction, labels)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', labels={labels}",
        )


class TestBallsimVerifier(unittest.IsolatedAsyncioTestCase):
    async def test_pass_rate_scoring(self):
        verifier = BallsimVerifier(
            BallsimVerifierConfig(
                ballsim_api_url="http://localhost:2345/test_program",
                ballsim_max_execution_time=1.0,
                ballsim_scoring_mode="pass_rate",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"results": [1, 0, 1], "runtimes": [0.1, 0.2, 0.1]}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(BallsimVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call([], "```python\npass\n```", ["assert True"], None)

        self.assertAlmostEqual(result.score, 2 / 3)

    async def test_all_pass_scoring(self):
        verifier = BallsimVerifier(
            BallsimVerifierConfig(
                ballsim_api_url="http://localhost:2345/test_program",
                ballsim_max_execution_time=1.0,
                ballsim_scoring_mode="all_pass",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"results": [1, 0, 1], "runtimes": [0.1, 0.2, 0.1]}
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(BallsimVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call([], "```python\npass\n```", ["assert True"], None)

        self.assertEqual(result.score, 0.0)


class TestManufactoriaVerifier(unittest.IsolatedAsyncioTestCase):
    async def test_pass_rate_scoring(self):
        verifier = ManufactoriaVerifier(
            ManufactoriaVerifierConfig(
                manufactoria_api_url="http://localhost:1235/test_solution",
                manufactoria_max_execution_time=1.0,
                manufactoria_scoring_mode="pass_rate",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "valid": True,
            "all_passed": False,
            "results": [{"passed": True}, {"passed": False}, {"passed": True}],
        }
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(ManufactoriaVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call(
                [], "```manufactoria\nSTART start:\n    NEXT end\nEND end\n```", [{}], None
            )

        self.assertAlmostEqual(result.score, 2 / 3)
        self.assertEqual(result.metadata.get("manufactoria_per_test_passed"), [1.0])

    async def test_all_pass_scoring(self):
        verifier = ManufactoriaVerifier(
            ManufactoriaVerifierConfig(
                manufactoria_api_url="http://localhost:1235/test_solution",
                manufactoria_max_execution_time=1.0,
                manufactoria_scoring_mode="all_pass",
            )
        )
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "valid": True,
            "all_passed": False,
            "results": [{"passed": True}, {"passed": False}, {"passed": True}],
        }
        mock_response.raise_for_status.return_value = None
        mock_session.post.return_value = mock_response

        with patch.object(ManufactoriaVerifier, "_get_session", return_value=mock_session):
            result = await verifier.async_call(
                [], "```manufactoria\nSTART start:\n    NEXT end\nEND end\n```", [{}], None
            )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.metadata.get("manufactoria_per_test_passed"), [1.0])


class TestGSM8KVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = GSM8KVerifier()

    @parameterized.expand(
        [
            ("negative_integer", "Therefore the answer is -3", "-3", 1.0),
            ("positive_integer", "Therefore the answer is +7", "+7", 1.0),
            ("negative_decimal", "Final answer: -3.5", "-3.5", 1.0),
            ("boxed_negative_integer", r"The result is \boxed{-3}", "-3", 1.0),
            ("wrong_sign", "Therefore the answer is 3", "-3", 0.0),
        ]
    )
    def test_signed_number_extraction(self, _name, prediction, label, expected_score):
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)


class TestRewardConfig(unittest.TestCase):
    def test_spurious_reward_mode_outputs_zero_or_verification_reward(self):
        verification_reward = 10
        reward_fn = RewardConfig(
            apply_r1_style_format_reward=False,
            apply_verifiable_reward=False,
            non_stop_penalty=False,
            spurious_reward_mode=True,
            verification_reward=verification_reward,
        ).build()
        n = 64
        scores, metrics = asyncio.run(
            reward_fn(
                responses=[[1, 2, 3] for _ in range(n)],
                decoded_responses=["x"] * n,
                ground_truths=["y"] * n,
                datasets=["gsm8k"] * n,
                finish_reasons=["stop"] * n,
                infos=RequestInfo(
                    num_calls=[0] * n,
                    timeouts=[0] * n,
                    tool_errors=[""] * n,
                    tool_outputs=[""] * n,
                    tool_runtimes=[0.0] * n,
                    tool_calleds=[False] * n,
                ),
                queries=["q"] * n,
            )
        )
        self.assertEqual(len(scores), n)
        self.assertTrue(all(score in {0.0, float(verification_reward)} for score in scores))
        self.assertIn("objective/spurious_reward", metrics)
        self.assertIn("objective/spurious_correct_rate", metrics)

    def test_spurious_reward_mode_logs_spurious_metrics(self):
        verification_reward = 10
        n = 4
        reward_fn = RewardConfig(
            apply_r1_style_format_reward=False,
            apply_verifiable_reward=True,
            non_stop_penalty=False,
            spurious_reward_mode=True,
            verification_reward=verification_reward,
            verifier_functions={},
        ).build()

        with (
            patch(
                "open_instruct.ground_truth_utils.apply_verifiable_reward",
                return_value=([0.0, 10.0, 10.0, 0.0], [{}, {}, {}, {}]),
            ),
            patch("open_instruct.ground_truth_utils.np.random.randint", return_value=np.array([1, 0, 1, 0])),
        ):
            scores, metrics = asyncio.run(
                reward_fn(
                    responses=[[1, 2, 3] for _ in range(n)],
                    decoded_responses=["x"] * n,
                    ground_truths=["y"] * n,
                    datasets=["gsm8k"] * n,
                    finish_reasons=["stop"] * n,
                    infos=RequestInfo(
                        num_calls=[0] * n,
                        timeouts=[0] * n,
                        tool_errors=[""] * n,
                        tool_outputs=[""] * n,
                        tool_runtimes=[0.0] * n,
                        tool_calleds=[False] * n,
                    ),
                    queries=["q"] * n,
                )
            )

        self.assertEqual(scores, [10.0, 0.0, 10.0, 0.0])
        self.assertNotIn("objective/true_objective_reward", metrics)
        self.assertNotIn("objective/true_objective_correct_rate", metrics)
        self.assertEqual(metrics["objective/spurious_reward"], 5.0)
        self.assertEqual(metrics["objective/spurious_correct_rate"], 0.5)


class TestApplyVerifiableRewardDatasetAliases(unittest.TestCase):
    class _DummyVerifier:
        def __init__(self, name: str, score: float = 1.0):
            self.name = name
            self.weight = 1.0
            self._score = score

        async def async_call(self, **kwargs):
            return SimpleNamespace(score=self._score)

    def test_math_prefixed_dataset_uses_math_verifier(self):
        verifier = self._DummyVerifier(name="math", score=1.0)
        scores, per_func_scores, extra_metrics = asyncio.run(
            apply_verifiable_reward(
                reward_fn_mapping={"math": verifier},
                responses=[[1, 2, 3]],
                decoded_responses=["dummy"],
                ground_truths=["42"],
                datasets=["math_hmmt_feb_2025"],
                reward_mult=10,
                queries=["q"],
            )
        )
        self.assertEqual(scores, [10.0])
        self.assertEqual(per_func_scores, [{"math": 10.0}])
        self.assertEqual(extra_metrics, {})

    class _StaticVerifier:
        def __init__(self, name: str, score: float):
            self.name = name
            self.weight = 1.0
            self._score = score

        def __call__(self, tokenized_prediction, prediction, label, query=None):
            return VerificationResult(score=self._score)

        async def async_call(self, tokenized_prediction, prediction, label, query=None):
            return VerificationResult(score=self._score)

    def test_llm_judge_fallback_logs_when_llm_overrides_primary_failure(self):
        primary = self._StaticVerifier(name="math", score=0.0)
        fallback = self._StaticVerifier(name="general-compass_verifier", score=1.0)
        fallback_wrapper = LLMJudgeFallbackVerifier(primary, fallback)

        scores, per_func_scores, extra_metrics = asyncio.run(
            apply_verifiable_reward(
                reward_fn_mapping={"math": fallback_wrapper},
                responses=[[1, 2, 3]],
                decoded_responses=["dummy"],
                ground_truths=["42"],
                datasets=["math"],
                reward_mult=10,
                queries=["q"],
            )
        )

        self.assertEqual(scores, [10.0])
        self.assertEqual(per_func_scores, [{"math": 10.0}])
        self.assertEqual(extra_metrics["objective/llm_judge_fallback_used_count"], 1.0)
        self.assertEqual(extra_metrics["objective/llm_judge_correct_when_primary_wrong_count"], 1.0)
        self.assertEqual(extra_metrics["objective/math_llm_judge_correct_when_primary_wrong_count"], 1.0)


class TestBuildAllVerifiers(unittest.TestCase):
    def test_llm_judge_override_verifier_replaces_gsm8k_with_compass_verifier(self):
        args = SimpleNamespace(
            llm_judge_model="opencompass/CompassVerifier-3B",
            llm_judge_max_tokens=256,
            llm_judge_max_context_length=8192,
            llm_judge_temperature=0.0,
            llm_judge_timeout=60,
            seed=1,
            code_api_url="http://localhost:1234/test_program",
            code_max_execution_time=1.0,
            code_pass_rate_reward_threshold=0.0,
            code_apply_perf_penalty=False,
            max_length_verifier_max_length=32768,
        )
        streaming_config = SimpleNamespace(
            llm_judge_model="opencompass/CompassVerifier-3B",
            llm_judge_override_verifier="gsm8k",
            llm_judge_max_tokens=256,
            llm_judge_max_context_length=8192,
            llm_judge_temperature=0.0,
            llm_judge_timeout=60,
            seed=1,
            remap_verifier=None,
        )

        verifiers = build_all_verifiers(args, streaming_config)
        self.assertIsInstance(verifiers["gsm8k"], LMJudgeVerifier)
        self.assertEqual(verifiers["gsm8k"].verifier_config.llm_judge_model, "opencompass/CompassVerifier-3B")
        self.assertEqual(verifiers["gsm8k"].judge_type, "compass_verifier")

    def test_llm_judge_fallback_verifier_wraps_math_with_compass_verifier(self):
        args = SimpleNamespace(
            llm_judge_model="opencompass/CompassVerifier-3B",
            llm_judge_max_tokens=256,
            llm_judge_max_context_length=8192,
            llm_judge_temperature=0.0,
            llm_judge_timeout=60,
            seed=1,
            code_api_url="http://localhost:1234/test_program",
            code_max_execution_time=1.0,
            code_pass_rate_reward_threshold=0.0,
            code_apply_perf_penalty=False,
            max_length_verifier_max_length=32768,
        )
        streaming_config = SimpleNamespace(
            llm_judge_model="opencompass/CompassVerifier-3B",
            llm_judge_override_verifier=None,
            llm_judge_fallback_verifier="math",
            llm_judge_max_tokens=256,
            llm_judge_max_context_length=8192,
            llm_judge_temperature=0.0,
            llm_judge_timeout=60,
            seed=1,
            remap_verifier=None,
        )

        verifiers = build_all_verifiers(args, streaming_config)
        self.assertIsInstance(verifiers["math"], LLMJudgeFallbackVerifier)
        self.assertEqual(verifiers["math"].fallback_verifier.judge_type, "compass_verifier")


class TestCompassVerifierExtractor(unittest.TestCase):
    @parameterized.expand(
        [
            ("plain_a", "A", 1.0),
            ("plain_b", "B", 0.0),
            ("plain_c", "C", 0.0),
            ("boxed", r"Final Judgment: \\boxed{A} - CORRECT", 1.0),
            ("incorrect_word", "INCORRECT", 0.0),
            ("invalid_word", "INVALID", 0.0),
        ]
    )
    def test_extract_score_compass_verifier(self, _name, text, expected_score):
        _, score = extract_score_compass_verifier(text)
        self.assertEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
