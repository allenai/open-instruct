#!/usr/bin/env python3
"""Unit tests for the rubrics module.

Tests for:
- rubric_utils.py: Evolving rubric generation and management
- run_utils.py: LiteLLM calls and JSON extraction
- metrics.py: Rubric metrics computation and buffer filtering
- RubricVerifier: Rubric-based scoring in ground_truth_utils.py

Usage:
    # From the repo root:
    uv run pytest open_instruct/rubrics/test_rubrics.py -v
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig
from open_instruct.rubrics.metrics import (
    compute_rubric_count_metrics,
    compute_rubric_reward_metrics,
    filter_rubric_buffer,
)
from open_instruct.rubrics.rubric_utils import (
    initialize_rubric_buffer,
    save_evolving_rubric_cache_safe,
    update_ground_truths_with_evolving_rubrics,
)
from open_instruct.rubrics.run_utils import extract_json_from_response


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


# =============================================================================
# Tests for run_utils.py
# =============================================================================


class TestExtractJsonFromResponse(unittest.TestCase):
    """Test suite for extract_json_from_response."""

    def test_basic_json(self):
        """Test basic JSON extraction."""
        self.assertEqual(extract_json_from_response('{"score": 2}'), {"score": 2})

    def test_json_with_text_before(self):
        """Test JSON with text before."""
        self.assertEqual(extract_json_from_response('The score is {"score": 1}'), {"score": 1})

    def test_json_with_text_after(self):
        """Test JSON with text after."""
        self.assertEqual(extract_json_from_response('{"score": 0} is the result'), {"score": 0})

    def test_json_in_markdown(self):
        """Test JSON in markdown code block."""
        self.assertEqual(extract_json_from_response('```json\n{"score": 2}\n```'), {"score": 2})

    def test_nested_json(self):
        """Test nested JSON."""
        self.assertEqual(extract_json_from_response('{"score": 1, "reason": "good"}'), {"score": 1, "reason": "good"})

    def test_double_braces(self):
        """Test double braces (common LLM mistake)."""
        self.assertEqual(extract_json_from_response('{{"score": 2}}'), {"score": 2})

    def test_no_json(self):
        """Test string with no JSON."""
        self.assertIsNone(extract_json_from_response("This has no JSON"))

    def test_invalid_json(self):
        """Test invalid JSON."""
        self.assertIsNone(extract_json_from_response('{"score": }'))

    def test_multiple_json_objects(self):
        """Test multiple JSON objects (should get last valid one)."""
        self.assertEqual(extract_json_from_response('{"a": 1} more text {"score": 2}'), {"score": 2})


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestRunLitellmAsync(unittest.TestCase):
    """Test suite for run_litellm_async (requires API)."""

    def test_async_call(self):
        """Test async LiteLLM calls."""
        from open_instruct.rubrics.run_utils import run_litellm_async  # noqa: PLC0415

        model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")

        response = asyncio.run(
            run_litellm_async(
                model_name=model, user_prompt="What is 2+2? Answer with just the number.", temperature=0, max_tokens=10
            )
        )
        self.assertIn(response.strip(), ["4", "4."])


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")), "No API credentials available"
)
class TestRunLitellmSync(unittest.TestCase):
    """Test suite for run_litellm (sync) (requires API)."""

    def test_sync_call(self):
        """Test sync LiteLLM calls."""
        from open_instruct.rubrics.run_utils import run_litellm  # noqa: PLC0415

        model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")

        response = run_litellm(
            model_name=model, user_prompt="What is 3+3? Answer with just the number.", temperature=0, max_tokens=10
        )
        self.assertIn(response.strip(), ["6", "6."])


# =============================================================================
# Tests for rubric_utils.py
# =============================================================================


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


# =============================================================================
# Tests for RubricVerifier (in ground_truth_utils.py)
# =============================================================================


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


# =============================================================================
# Tests for metrics.py
# =============================================================================


class TestComputeRubricRewardMetrics(unittest.TestCase):
    """Test suite for compute_rubric_reward_metrics."""

    def test_split_by_type(self):
        """Test that rewards are correctly split by rubric type."""
        per_rubric_scores = [
            [(0.8, 1.0), (0.6, 1.0), (0.4, -1.0)],  # response 1
            [(0.5, 1.0), (0.9, 1.0)],  # response 2
        ]
        per_rubric_types = [["persistent", "evolving", "evolving"], ["persistent", "evolving"]]
        metrics = compute_rubric_reward_metrics(per_rubric_scores, per_rubric_types)
        self.assertIn("evolving_rubrics/avg_evolving_reward", metrics)
        self.assertIn("evolving_rubrics/std_evolving_reward", metrics)
        self.assertIn("evolving_rubrics/avg_persistent_reward", metrics)
        self.assertIn("evolving_rubrics/std_persistent_reward", metrics)
        # persistent: 0.8*1.0, 0.5*1.0 → mean = 0.65
        self.assertAlmostEqual(metrics["evolving_rubrics/avg_persistent_reward"], 0.65)
        self.assertGreater(metrics["evolving_rubrics/std_persistent_reward"], 0.0)
        # evolving: 0.6*1.0, 0.4*(-1.0), 0.9*1.0 → mean of [0.6, -0.4, 0.9] = 1.1/3
        self.assertAlmostEqual(metrics["evolving_rubrics/avg_evolving_reward"], 1.1 / 3, places=5)
        self.assertGreater(metrics["evolving_rubrics/std_evolving_reward"], 0.0)

    def test_empty_scores(self):
        """Test with no rubric scores."""
        metrics = compute_rubric_reward_metrics([], [])
        self.assertEqual(metrics["evolving_rubrics/avg_evolving_reward"], 0.0)
        self.assertEqual(metrics["evolving_rubrics/std_evolving_reward"], 0.0)
        self.assertEqual(metrics["evolving_rubrics/avg_persistent_reward"], 0.0)
        self.assertEqual(metrics["evolving_rubrics/std_persistent_reward"], 0.0)

    def test_only_persistent(self):
        """Test with only persistent rubrics."""
        per_rubric_scores = [[(0.7, 1.0)]]
        per_rubric_types = [["persistent"]]
        metrics = compute_rubric_reward_metrics(per_rubric_scores, per_rubric_types)
        self.assertAlmostEqual(metrics["evolving_rubrics/avg_persistent_reward"], 0.7)
        self.assertEqual(metrics["evolving_rubrics/std_persistent_reward"], 0.0)
        self.assertEqual(metrics["evolving_rubrics/avg_evolving_reward"], 0.0)
        self.assertEqual(metrics["evolving_rubrics/std_evolving_reward"], 0.0)


class TestComputeRubricCountMetrics(unittest.TestCase):
    """Test suite for compute_rubric_count_metrics."""

    def test_basic_counts(self):
        """Test that count metrics are produced correctly."""
        metrics = compute_rubric_count_metrics(avg_num_evolving_rubrics=2.5, avg_num_active_buffer_rubrics=4.0)
        self.assertEqual(metrics["evolving_rubrics/num_new_rubrics"], 2.5)
        self.assertEqual(metrics["evolving_rubrics/num_active_rubrics"], 4.0)

    def test_zero_counts(self):
        """Test with zero counts."""
        metrics = compute_rubric_count_metrics(0.0, 0.0)
        self.assertEqual(len(metrics), 2)
        for v in metrics.values():
            self.assertEqual(v, 0.0)


class TestFilterRubricBuffer(unittest.TestCase):
    """Test suite for filter_rubric_buffer."""

    def _make_buffer(self):
        """Create a test rubric buffer."""
        return {
            "Q1": {
                "persistent_rubrics": [{"description": "persistent1", "weight": 1.0, "title": "P1"}],
                "active_rubrics": [
                    {"description": "evolving1", "weight": 1.0, "title": "E1"},
                    {"description": "evolving2", "weight": 1.0, "title": "E2"},
                    {"description": "evolving3", "weight": 1.0, "title": "E3"},
                ],
                "inactive_rubrics": [],
            }
        }

    def test_deactivate_zero_std(self):
        """Test that zero-std rubrics are moved to inactive."""
        buffer = self._make_buffer()
        stats = {
            "Q1::E1": {"mean": 0.5, "std": 0.3},
            "Q1::E2": {"mean": 0.2, "std": 0.0},  # zero std → deactivate
            "Q1::E3": {"mean": 0.8, "std": 0.1},
        }
        filter_rubric_buffer(buffer, stats, max_active_rubrics=5)
        self.assertEqual(len(buffer["Q1"]["active_rubrics"]), 2)
        self.assertEqual(len(buffer["Q1"]["inactive_rubrics"]), 1)
        self.assertEqual(buffer["Q1"]["inactive_rubrics"][0]["title"], "E2")

    def test_cap_active_rubrics(self):
        """Test that excess active rubrics are capped."""
        buffer = self._make_buffer()
        stats = {
            "Q1::E1": {"mean": 0.5, "std": 0.3},
            "Q1::E2": {"mean": 0.5, "std": 0.5},
            "Q1::E3": {"mean": 0.5, "std": 0.1},
        }
        filter_rubric_buffer(buffer, stats, max_active_rubrics=2)
        # Should keep E2 (std=0.5) and E1 (std=0.3), move E3 (std=0.1)
        self.assertLessEqual(len(buffer["Q1"]["active_rubrics"]), 2)

    def test_empty_buffer(self):
        """Test with empty buffer — no crash."""
        buffer: dict = {}
        stats = {"Q1::E1": {"mean": 0.5, "std": 0.3}}
        filter_rubric_buffer(buffer, stats, max_active_rubrics=5)
        # No assertions needed — just verifying it doesn't crash

    def test_no_changes_needed(self):
        """Test when all rubrics have good std and are under cap."""
        buffer = self._make_buffer()
        stats = {
            "Q1::E1": {"mean": 0.5, "std": 0.3},
            "Q1::E2": {"mean": 0.5, "std": 0.5},
            "Q1::E3": {"mean": 0.5, "std": 0.1},
        }
        filter_rubric_buffer(buffer, stats, max_active_rubrics=10)
        self.assertEqual(len(buffer["Q1"]["active_rubrics"]), 3)
        self.assertEqual(len(buffer["Q1"]["inactive_rubrics"]), 0)


if __name__ == "__main__":
    unittest.main()
