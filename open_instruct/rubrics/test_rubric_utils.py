#!/usr/bin/env python3
"""Test script for rubric_utils.py - tests evolving rubric generation functions.

Usage:
    # From the repo root:
    uv run pytest open_instruct/rubrics/test_rubric_utils.py -v
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

from open_instruct.rubrics.rubric_utils import (
    initialize_rubric_buffer,
    save_evolving_rubric_cache_safe,
    update_ground_truths_with_evolving_rubrics,
)


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class TestInitializeRubricBuffer(unittest.TestCase):
    """Test suite for initialize_rubric_buffer."""

    def test_buffer_structure(self):
        """Test rubric buffer initialization with static rubrics as persistent."""
        ground_truths = [
            json.dumps(
                {
                    "query": "What is the capital of France?",
                    "rubrics": [
                        {"description": "Correctly identifies Paris", "weight": 1.0},
                        {"description": "Provides context about France", "weight": 0.5},
                    ],
                }
            ),
            json.dumps(
                {
                    "query": "Explain photosynthesis.",
                    "rubrics": [
                        {"description": "Mentions chlorophyll", "weight": 1.0},
                        {"description": "Explains light absorption", "weight": 1.0},
                    ],
                }
            ),
        ]

        buffer = initialize_rubric_buffer(ground_truths)

        self.assertEqual(len(buffer), 2)
        self.assertIn("What is the capital of France?", buffer)
        self.assertIn("Explain photosynthesis.", buffer)

        france_entry = buffer["What is the capital of France?"]
        self.assertIn("active_rubrics", france_entry)
        self.assertIn("inactive_rubrics", france_entry)
        self.assertIn("persistent_rubrics", france_entry)

        # Static rubrics become persistent, active starts empty
        self.assertEqual(len(france_entry["active_rubrics"]), 0)
        self.assertEqual(len(france_entry["persistent_rubrics"]), 2)


class TestUpdateGroundTruthsWithEvolvingRubrics(unittest.TestCase):
    """Test suite for update_ground_truths_with_evolving_rubrics."""

    def test_evolving_rubrics_added_correctly(self):
        """Test that evolving rubrics are added to ground truths correctly."""
        ground_truths = [
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
        ]

        all_evolving_rubrics = [
            {
                "question": "Q1",
                "positive_rubrics": [{"description": "Positive1", "title": "P1"}],
                "negative_rubrics": [{"description": "Negative1", "title": "N1"}],
            },
            {
                "question": "Q2",
                "positive_rubrics": [{"description": "Positive2", "title": "P2"}],
                "negative_rubrics": [],
            },
        ]

        num_samples_per_prompt_rollout = 2

        updated_gts, rate, avg_gt, avg_ar, avg_active_buf, buffer, skipped = (
            update_ground_truths_with_evolving_rubrics(
                ground_truths.copy(), all_evolving_rubrics, num_samples_per_prompt_rollout, rubric_buffer=None
            )
        )

        self.assertEqual(len(updated_gts), 4)

        gt1 = json.loads(updated_gts[0][0])
        self.assertEqual(len(gt1["rubrics"]), 3)  # 1 original + 2 evolving
        self.assertEqual(gt1["rubrics"][0]["description"], "R1")
        self.assertEqual(gt1["rubrics"][1]["description"], "Positive1")
        self.assertEqual(gt1["rubrics"][1]["weight"], 1.0)
        self.assertEqual(gt1["rubrics"][2]["description"], "Negative1")
        self.assertEqual(gt1["rubrics"][2]["weight"], -1.0)

    def test_second_prompt_rubrics_correct(self):
        """Test that second prompt rubrics are updated correctly."""
        ground_truths = [
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
        ]

        all_evolving_rubrics = [
            {
                "question": "Q1",
                "positive_rubrics": [{"description": "Positive1", "title": "P1"}],
                "negative_rubrics": [{"description": "Negative1", "title": "N1"}],
            },
            {
                "question": "Q2",
                "positive_rubrics": [{"description": "Positive2", "title": "P2"}],
                "negative_rubrics": [],
            },
        ]

        updated_gts, *_ = update_ground_truths_with_evolving_rubrics(
            ground_truths.copy(), all_evolving_rubrics, 2, rubric_buffer=None
        )

        gt3 = json.loads(updated_gts[2][0])
        self.assertEqual(len(gt3["rubrics"]), 2)

    def test_handles_none_rubrics(self):
        """Test that None evolving rubrics are handled correctly."""
        ground_truths = [
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
            [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
        ]

        evolving_with_none = [
            None,
            {
                "question": "Q2",
                "positive_rubrics": [{"description": "Positive2", "title": "P2"}],
                "negative_rubrics": [],
            },
        ]

        _, _, _, _, _, _, skipped = update_ground_truths_with_evolving_rubrics(
            ground_truths.copy(), evolving_with_none, 2, rubric_buffer=None
        )

        self.assertEqual(skipped, 2)  # None affects 2 samples


class TestSaveEvolvingRubricCacheSafe(unittest.TestCase):
    """Test suite for save_evolving_rubric_cache_safe."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_dir = tempfile.mkdtemp()
        self.test_data = {
            "decoded_responses": ["Response 1", "Response 2"],
            "ground_truths": [json.dumps({"query": "Q1", "rubrics": []}), json.dumps({"query": "Q2", "rubrics": []})],
            "all_evolving_rubrics": [
                {"positive_rubrics": [], "negative_rubrics": []},
                {"positive_rubrics": [], "negative_rubrics": []},
            ],
            "num_subsampled_answers_list": [2, 2],
            "num_samples_per_prompt_rollout": 2,
            "use_full_responses": True,
            "answer_length_limit_in_words": None,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.cache_dir)

    def test_cache_file_created(self):
        """Test that cache file is created."""
        cache_path = save_evolving_rubric_cache_safe(cache_dir=self.cache_dir, training_step=1, **self.test_data)
        self.assertTrue(os.path.exists(cache_path))

    def test_cache_contents_correct(self):
        """Test that cache contents are correct."""
        cache_path = save_evolving_rubric_cache_safe(cache_dir=self.cache_dir, training_step=1, **self.test_data)

        with open(cache_path) as f:
            loaded_data = json.load(f)

        self.assertEqual(loaded_data["training_step"], 1)
        self.assertEqual(loaded_data["inputs"]["decoded_responses"], self.test_data["decoded_responses"])
        self.assertEqual(loaded_data["outputs"]["all_evolving_rubrics"], self.test_data["all_evolving_rubrics"])

    def test_multiple_saves_unique_files(self):
        """Test that multiple saves create unique files."""
        cache_path1 = save_evolving_rubric_cache_safe(cache_dir=self.cache_dir, training_step=1, **self.test_data)
        cache_path2 = save_evolving_rubric_cache_safe(cache_dir=self.cache_dir, training_step=2, **self.test_data)

        self.assertTrue(os.path.exists(cache_path2))
        self.assertNotEqual(cache_path1, cache_path2)


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestGenerateInstanceWiseEvolvingRubrics(unittest.TestCase):
    """Test suite for generate_instance_wise_evolving_rubrics (requires API)."""

    def test_generate_rubrics(self):
        """Test evolving rubric generation with real API calls."""
        from open_instruct.rubrics.rubric_utils import generate_instance_wise_evolving_rubrics  # noqa: PLC0415

        question = "What is the capital of France?"
        response_list = [
            "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            "Paris is the capital.",
            "France's capital city is Paris, which has been the capital since the 10th century.",
        ]
        existing_rubrics = json.dumps([{"description": "Correctly identifies Paris as the capital", "weight": 1.0}])

        model = os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1-mini")

        result = asyncio.run(
            generate_instance_wise_evolving_rubrics(
                question=question, response_list=response_list, existing_rubrics=existing_rubrics, model_name=model
            )
        )

        self.assertIsNotNone(result)
        self.assertTrue("positive_rubrics" in result or "negative_rubrics" in result)


if __name__ == "__main__":
    unittest.main()
