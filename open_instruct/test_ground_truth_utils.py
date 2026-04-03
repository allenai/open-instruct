#!/usr/bin/env python3
"""
Test script for verifier functionality in Python
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from parameterized import parameterized

from open_instruct.ground_truth_utils import (
    F1Verifier,
    GSM8KVerifier,
    LMJudgeVerifier,
    LMJudgeVerifierConfig,
    PuzzleMatcherVerifier,
    cleanup_all_llm_judge_clients,
)


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


class TestGSM8KVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = GSM8KVerifier()

    @parameterized.expand(
        [
            ("negative_integer", "Therefore the answer is -3", "-3", 1.0),
            ("positive_integer", "Therefore the answer is +7", "+7", 1.0),
            ("negative_decimal", "Final answer: -3.5", "-3.5", 1.0),
            ("boxed_negative_integer", r"The result is \\boxed{-3}", "-3", 1.0),
            ("wrong_sign", "Therefore the answer is 3", "-3", 0.0),
        ]
    )
    def test_signed_number_extraction(self, _name, prediction, label, expected_score):
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)


def _make_litellm_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class TestLMJudgeVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = LMJudgeVerifier(
            "quality",
            LMJudgeVerifierConfig(
                llm_judge_model="azure/gpt-4o-mini-standard",
                llm_judge_max_tokens=256,
                llm_judge_max_context_length=4096,
                llm_judge_temperature=0.0,
                llm_judge_timeout=30,
                seed=17,
            ),
        )

    def test_async_call_uses_shared_helper_and_preserves_retry_and_cost(self):
        response = _make_litellm_response('{"REASONING":"clear","SCORE":7}', prompt_tokens=10, completion_tokens=5)
        raw_helper = AsyncMock(side_effect=[RuntimeError("temporary"), response])
        sleep_mock = AsyncMock()

        with (
            patch("open_instruct.ground_truth_utils.run_litellm_async_raw", raw_helper),
            patch(
                "open_instruct.ground_truth_utils.context_window_checker.check_context_window_limit", return_value=True
            ),
            patch("open_instruct.ground_truth_utils.asyncio.sleep", sleep_mock),
        ):
            result = asyncio.run(
                self.verifier.async_call(
                    tokenized_prediction=[],
                    prediction="<answer>final answer</answer>",
                    label="reference",
                    query="What is the answer?",
                )
            )

        self.assertAlmostEqual(result.score, 0.7)
        self.assertEqual(result.reasoning, "clear")
        self.assertAlmostEqual(result.cost, 0.0000045)
        self.assertEqual(raw_helper.await_count, 2)
        self.assertEqual(sleep_mock.await_count, 1)
        self.assertEqual(raw_helper.await_args_list[-1].kwargs["model_name"], "azure/gpt-4o-mini-standard")
        self.assertEqual(raw_helper.await_args_list[-1].kwargs["max_completion_tokens"], 256)
        self.assertNotIn("num_retries", raw_helper.await_args_list[-1].kwargs)
        self.assertNotIn("fallbacks", raw_helper.await_args_list[-1].kwargs)

    def test_cleanup_helpers_are_safe_noops(self):
        self.assertIsNone(asyncio.run(LMJudgeVerifier.cleanup_all_clients()))
        self.assertIsNone(asyncio.run(cleanup_all_llm_judge_clients()))


if __name__ == "__main__":
    unittest.main()
