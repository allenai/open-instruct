#!/usr/bin/env python3
"""
Test script for verifier functionality in Python
"""

import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nltk
from parameterized import parameterized

from open_instruct import ground_truth_utils
from open_instruct.ground_truth_utils import (
    F1Verifier,
    GSM8KVerifier,
    LMJudgeVerifier,
    LMJudgeVerifierConfig,
    PuzzleMatcherVerifier,
    RewardConfig,
    cleanup_all_llm_judge_clients,
)
from open_instruct.IFEvalG import instructions_registry


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

    @parameterized.expand(
        [
            ("trailing_zero_decimal", "The total cost is $96.00", "96", 1.0),
            ("decimal_label_integer_answer", "The answer is 42", "42.0", 1.0),
            ("thousands_separator_in_answer", "It takes 1,350 minutes", "1350", 1.0),
            ("thousands_separator_in_label", "It takes 1350 minutes", "1,350", 1.0),
            ("different_value", "The total cost is $96.50", "96", 0.0),
            ("rounding_is_not_equality", "About 0.33 of them", "0.3333", 0.0),
            ("non_numeric_label_uses_string_match", "The answer is 7", "seven", 0.0),
        ]
    )
    def test_numeric_equivalence(self, _name, prediction, label, expected_score):
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)


class TestR1StyleFormatRewardGate(unittest.TestCase):
    """Regression tests for the non-additive R1-style format gate in RewardConfig.build().

    The gate used to compare format_scores[i] to the literal 1, so any
    r1_style_format_reward other than the default 1.0 zeroed the verifiable
    reward for every response, formatted or not."""

    FORMATTED_CORRECT = "<think>let me work it out</think> <answer>42</answer>"
    UNFORMATTED_CORRECT = "The answer is 42"
    FORMATTED_WRONG = "<think>let me work it out</think> <answer>41</answer>"

    def _run_reward_fn(self, scale: float, responses: list[str], additive: bool = False) -> list[float]:
        config = RewardConfig(
            apply_r1_style_format_reward=True,
            r1_style_format_reward=scale,
            apply_verifiable_reward=True,
            verification_reward=10.0,
            additive_format_reward=additive,
            verifier_functions={"gsm8k": GSM8KVerifier()},
        )
        reward_fn = config.build()
        infos = SimpleNamespace(
            timeouts=[False] * len(responses),
            tool_errors=[""] * len(responses),
            tool_outputs=[""] * len(responses),
            tool_calleds=[False] * len(responses),
            rollout_states=[{} for _ in responses],
        )
        scores, _metrics = asyncio.run(
            reward_fn(
                responses=[[] for _ in responses],
                decoded_responses=responses,
                ground_truths=["42"] * len(responses),
                datasets=["gsm8k"] * len(responses),
                finish_reasons=["stop"] * len(responses),
                infos=infos,
            )
        )
        return scores

    @parameterized.expand([("default_scale", 1.0), ("half_scale", 0.5), ("double_scale", 2.0)])
    def test_formatted_correct_answer_keeps_verifiable_reward(self, _name, scale):
        scores = self._run_reward_fn(scale, [self.FORMATTED_CORRECT])
        self.assertEqual(scores, [10.0])

    def test_unformatted_correct_answer_is_gated_to_zero_with_non_default_scale(self):
        scores = self._run_reward_fn(0.5, [self.UNFORMATTED_CORRECT])
        self.assertEqual(scores, [0.0])

    def test_formatted_wrong_answer_scores_zero_with_non_default_scale(self):
        scores = self._run_reward_fn(0.5, [self.FORMATTED_WRONG])
        self.assertEqual(scores, [0.0])

    def test_mixed_batch_with_non_default_scale(self):
        scores = self._run_reward_fn(0.5, [self.FORMATTED_CORRECT, self.UNFORMATTED_CORRECT, self.FORMATTED_WRONG])
        self.assertEqual(scores, [10.0, 0.0, 0.0])

    def test_additive_format_reward_with_non_default_scale(self):
        scores = self._run_reward_fn(0.5, [self.FORMATTED_CORRECT], additive=True)
        self.assertEqual(scores, [10.5])


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


class TestIFEvalVerifierEmptyInstructions(unittest.TestCase):
    """Regression test for PR #1655: IFEvalVerifier crashed with
    ZeroDivisionError when the constraint's instruction_id list was empty."""

    def test_empty_instruction_list_returns_zero_score(self):
        verifier = ground_truth_utils.IFEvalVerifier()
        label = str([{"instruction_id": [], "kwargs": []}])
        result = verifier(tokenized_prediction=[1, 2, 3], prediction="some non-empty response", label=label)
        self.assertEqual(result.score, 0.0)


class TestIFEvalGJsonFormatDeepNesting(unittest.TestCase):
    """Regression test: detectable_format:json_format called json.loads on the response and caught only ValueError,
    so a degenerate, deeply nested response raised RecursionError out of the verifier and stopped an RL run."""

    def test_deeply_nested_response_fails_the_constraint(self):
        checker = instructions_registry.INSTRUCTION_DICT["detectable_format:json_format"](
            "detectable_format:json_format"
        )
        checker.build_description()
        response = "[" * 100_000 + "]" * 100_000
        with self.assertRaises(RecursionError):
            json.loads(response)
        self.assertFalse(checker.check_following(response))
        self.assertTrue(checker.check_following('```json\n{"a": [1, 2]}\n```'))

    def test_legacy_json_verifier_rejects_deep_nesting(self):
        verifier = ground_truth_utils.IFEvalVerifierOld()
        response = "[" * 100_000 + "]" * 100_000
        self.assertEqual(verifier([], response, {"func_name": "validate_json_format"}).score, 0.0)
        self.assertEqual(verifier([], '{"a": [1, 2]}', {"func_name": "validate_json_format"}).score, 1.0)
        self.assertEqual(verifier([], "not JSON", {"func_name": "validate_json_format"}).score, 0.0)


class TestIFEvalGDegenerateConstraints(unittest.TestCase):
    """Regression tests for constraints that rewarded degenerate answers (marin-community/marin#10320).

    count:lowercase_counting capped the total number of lowercase words at N (2-3), so only an empty, all-caps
    or non-Latin answer passed; letters:letter_counting counted only ASCII letters, so a non-Latin answer met
    "less than N letters"; length_constraints:nth_paragraph_first_word redrew an out-of-range paragraph index
    at random on every check."""

    @staticmethod
    def _checker(key, **kwargs):
        checker = instructions_registry.INSTRUCTION_DICT[key](key)
        checker.build_description(**kwargs)
        return checker

    def test_lowercase_counting_caps_each_word(self):
        checker = self._checker("count:lowercase_counting", N=2)
        self.assertTrue(checker.check_following("The cat sat on a mat, then the dog sat."))
        self.assertFalse(checker.check_following("the cat and the dog and the bird"))
        self.assertTrue(checker.check_following(""))

    def test_letter_counting_counts_every_script(self):
        less = self._checker("letters:letter_counting", N=3, relation="less than")
        self.assertTrue(less.check_following("42"))
        self.assertFalse(less.check_following("答案是四十二"))
        self.assertFalse(less.check_following("Forty-two"))
        at_least = self._checker("letters:letter_counting", N=3, relation="at least")
        self.assertTrue(at_least.check_following("答案是四十二"))

    def test_out_of_range_paragraph_is_kept_and_fails(self):
        key = "length_constraints:nth_paragraph_first_word"
        response = "\n\n".join(["guest one"] * 7)
        for _ in range(50):
            checker = self._checker(key, num_paragraphs=7, nth_paragraph=8, first_word="guest")
            self.assertEqual(checker.get_instruction_args()["nth_paragraph"], 8)
            self.assertFalse(checker.check_following(response))
        self.assertTrue(
            self._checker(key, num_paragraphs=7, nth_paragraph=7, first_word="guest").check_following(response)
        )

    def test_random_paragraph_index_is_in_range(self):
        for _ in range(200):
            args = self._checker(
                "length_constraints:nth_paragraph_first_word", num_paragraphs=3
            ).get_instruction_args()
            self.assertLessEqual(args["nth_paragraph"], 3)


class TestIFEvalGWordCounting(unittest.TestCase):
    """Regression tests: `count:counting_composition` and `count:count_unique` used
    nltk.word_tokenize, which emits punctuation (".", ",") as standalone tokens. That
    made "exactly N words per sentence" off by one and any multi-sentence response look
    like it had duplicate words. The checkers now count only word tokens (instructions_util.word_tokens)."""

    @classmethod
    def setUpClass(cls):
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            resource_dir = tempfile.TemporaryDirectory(prefix="ifevalg-nltk-")
            cls.addClassCleanup(resource_dir.cleanup)
            nltk.download("punkt_tab", download_dir=resource_dir.name, quiet=True, raise_on_error=True)
            nltk.data.path.insert(0, resource_dir.name)
            cls.addClassCleanup(nltk.data.path.remove, resource_dir.name)

    def setUp(self):
        self.cc = instructions_registry.INSTRUCTION_DICT["count:counting_composition"]("count:counting_composition")
        self.cu = instructions_registry.INSTRUCTION_DICT["count:count_unique"]("count:count_unique")

    def test_counting_composition_compliant_with_spaced_divider(self):
        # 3 paragraphs, 3 sentences each, 2 words each, using the "* * *" divider the prompt asks for.
        self.cc.build_description(n_sent=3, n_words=2)
        response = (
            "Cats run. Dogs jump. Birds fly.\n\n* * *\n\n"
            "Fish swim. Foxes leap. Owls hoot.\n\n* * *\n\n"
            "Apes climb. Bats hang. Eels glide."
        )
        self.assertTrue(self.cc.check_following(response))

    def test_counting_composition_reported_example_and_whitespace(self):
        self.cc.build_description(n_sent=2, n_words=2)
        for divider in ("\n* * *\n", "\n  *  *\t*  \n", "\r\n* * *\r\n"):
            with self.subTest(divider=divider):
                response = divider.join(["Cats sleep. Dogs run."] * 3)
                self.assertTrue(self.cc.check_following(response))

    def test_counting_composition_accepts_outer_dividers(self):
        self.cc.build_description(n_sent=2, n_words=2)
        response = "\n* * *\n".join(["", *["Cats sleep. Dogs run."] * 3, ""])
        self.assertTrue(self.cc.check_following(response))

    def test_counting_composition_rejects_inline_or_vertical_dividers(self):
        self.cc.build_description(n_sent=2, n_words=2)
        for divider in (" * * * ", "\n*\n*\n*\n"):
            with self.subTest(divider=divider):
                response = divider.join(["Cats sleep. Dogs run."] * 3)
                self.assertFalse(self.cc.check_following(response))

    def test_counting_composition_rejects_wrong_structure(self):
        self.cc.build_description(n_sent=2, n_words=2)
        for paragraphs in (
            ["Cats sleep. Dogs run."] * 2,
            ["Cats sleep. Dogs run."] * 4,
            ["Cats sleep."] * 3,
            ["Cats sleep. Dogs run.", "", "Cats sleep. Dogs run."],
        ):
            with self.subTest(paragraphs=paragraphs):
                self.assertFalse(self.cc.check_following("\n* * *\n".join(paragraphs)))

    def test_counting_composition_via_ifeval_verifier(self):
        verifier = ground_truth_utils.IFEvalVerifier()
        label = str([{"instruction_id": ["count:counting_composition"], "kwargs": [{"n_sent": 2, "n_words": 2}]}])
        result = verifier(
            tokenized_prediction=[1, 2, 3], prediction="\n* * *\n".join(["Cats sleep. Dogs run."] * 3), label=label
        )
        self.assertEqual(result.score, 1.0)

    def test_counting_composition_rejects_spaceless_divider(self):
        # The instruction specifies the "* * *" divider; a "***" response does not follow it.
        self.cc.build_description(n_sent=3, n_words=2)
        response = (
            "Cats run. Dogs jump. Birds fly.\n\n***\n\n"
            "Fish swim. Foxes leap. Owls hoot.\n\n***\n\n"
            "Apes climb. Bats hang. Eels glide."
        )
        self.assertFalse(self.cc.check_following(response))

    def test_counting_composition_rejects_wrong_word_count(self):
        self.cc.build_description(n_sent=3, n_words=2)
        response = (
            "Cats run fast. Dogs jump. Birds fly.\n\n* * *\n\n"  # first sentence has 3 words
            "Fish swim. Foxes leap. Owls hoot.\n\n* * *\n\n"
            "Apes climb. Bats hang. Eels glide."
        )
        self.assertFalse(self.cc.check_following(response))

    def test_count_unique_allows_unique_multi_sentence(self):
        self.cu.build_description()
        self.assertTrue(self.cu.check_following("Alpha beta gamma. Delta epsilon zeta."))

    def test_count_unique_rejects_repeated_word(self):
        self.cu.build_description()
        self.assertFalse(self.cu.check_following("The sun is hot. The moon is cold."))  # "The"/"is" repeat

    def test_count_unique_via_ifeval_verifier(self):
        verifier = ground_truth_utils.IFEvalVerifier()
        label = str([{"instruction_id": ["count:count_unique"], "kwargs": [None]}])
        result = verifier(
            tokenized_prediction=[1, 2, 3], prediction="Alpha beta gamma. Delta epsilon zeta.", label=label
        )
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
