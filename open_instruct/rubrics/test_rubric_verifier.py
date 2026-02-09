#!/usr/bin/env python3
"""Test script for RubricVerifier in ground_truth_utils.py.

Usage:
    # From the repo root:
    uv run pytest open_instruct/rubrics/test_rubric_verifier.py -v
"""

import asyncio
import json
import os
import unittest

from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class TestRubricVerifierConfig(unittest.TestCase):
    """Test suite for RubricVerifierConfig."""

    def test_default_config_values(self):
        """Test default config values are correct."""
        config = RubricVerifierConfig()
        self.assertEqual(config.rubric_judge_model, "gpt-4.1")
        self.assertEqual(config.rubric_judge_max_tokens, 2048)
        self.assertEqual(config.rubric_judge_temperature, 0.0)

    def test_custom_config_values(self):
        """Test custom config values are set correctly."""
        config = RubricVerifierConfig(
            rubric_judge_model="gpt-4.1-mini", rubric_judge_max_tokens=1024, rubric_judge_temperature=0.5
        )
        self.assertEqual(config.rubric_judge_model, "gpt-4.1-mini")
        self.assertEqual(config.rubric_judge_max_tokens, 1024)


class TestRubricVerifierCreation(unittest.TestCase):
    """Test suite for RubricVerifier instantiation."""

    def test_verifier_created_successfully(self):
        """Test verifier can be created."""
        config = RubricVerifierConfig()
        verifier = RubricVerifier(config)

        self.assertEqual(verifier.name, "rubric")
        self.assertEqual(verifier.weight, 1.0)
        self.assertEqual(verifier.verifier_config, config)

    def test_get_config_class(self):
        """Test get_config_class returns correct class."""
        self.assertEqual(RubricVerifier.get_config_class(), RubricVerifierConfig)


class TestRubricVerifierEdgeCases(unittest.TestCase):
    """Test suite for RubricVerifier edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RubricVerifierConfig()
        self.verifier = RubricVerifier(self.config)

    def test_empty_rubrics(self):
        """Test empty rubrics return score 0."""
        label_empty = json.dumps({"query": "Test question", "rubrics": []})
        result = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[], prediction="Some response", label=label_empty, query="Test question"
            )
        )
        self.assertEqual(result.score, 0.0)

    def test_invalid_json_label(self):
        """Test invalid JSON returns score 0."""
        result = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[], prediction="Some response", label="not valid json", query="Test question"
            )
        )
        self.assertEqual(result.score, 0.0)

    def test_non_dict_label(self):
        """Test non-dict label returns score 0."""
        result = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction="Some response",
                label=json.dumps(["not", "a", "dict"]),
                query="Test question",
            )
        )
        self.assertEqual(result.score, 0.0)


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestRubricVerifierScoring(unittest.TestCase):
    """Test suite for RubricVerifier scoring (requires API)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RubricVerifierConfig(rubric_judge_model=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini"))
        self.verifier = RubricVerifier(self.config)

    def test_good_response_scores_high(self):
        """Test good response gets high score."""
        prediction_good = "The capital of France is Paris. Paris is a major European city."
        label_good = json.dumps(
            {
                "query": "What is the capital of France?",
                "rubrics": [{"description": "Correctly identifies Paris as the capital of France", "weight": 1.0}],
            }
        )

        result = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction=prediction_good,
                label=label_good,
                query="What is the capital of France?",
            )
        )

        self.assertGreaterEqual(result.score, 0.5)

    def test_bad_response_scores_low(self):
        """Test bad response gets lower score than good response."""
        label = json.dumps(
            {
                "query": "What is the capital of France?",
                "rubrics": [{"description": "Correctly identifies Paris as the capital of France", "weight": 1.0}],
            }
        )

        result_good = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction="The capital of France is Paris.",
                label=label,
                query="What is the capital of France?",
            )
        )

        result_bad = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction="The capital of France is London.",
                label=label,
                query="What is the capital of France?",
            )
        )

        self.assertLess(result_bad.score, result_good.score)

    def test_multi_rubric_scoring(self):
        """Test scoring with multiple rubrics works."""
        label_multi = json.dumps(
            {
                "query": "Explain photosynthesis briefly.",
                "rubrics": [
                    {"description": "Mentions that plants use sunlight", "weight": 1.0},
                    {"description": "Mentions carbon dioxide", "weight": 0.5},
                    {"description": "Mentions oxygen production", "weight": 0.5},
                ],
            }
        )
        prediction = "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen."

        result = asyncio.run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction=prediction,
                label=label_multi,
                query="Explain photosynthesis briefly.",
            )
        )

        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 1)


if __name__ == "__main__":
    unittest.main()
