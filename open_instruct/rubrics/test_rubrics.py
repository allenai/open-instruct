#!/usr/bin/env python3
"""Unit tests for the rubrics module.

Tests for:
- rubric_utils.py: Evolving rubric generation and management
- run_utils.py: LiteLLM calls and JSON extraction
- metrics.py: Rubric metrics computation and buffer filtering
- RubricVerifier: Rubric-based scoring in ground_truth_utils.py
- evolving_rubric_step.py: End-to-end evolving rubric step wiring

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
from typing import Any
from unittest.mock import MagicMock, patch

from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig
from open_instruct.model_utils import Batch
from open_instruct.rubrics.evolving_rubric_step import (
    EvolvingRubricConfig,
    _push_ground_truth_overrides,
    init_rubric_buffer,
    run_evolving_rubric_step,
)
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


# =============================================================================
# Tests for evolving_rubric_step.py (end-to-end wiring)
# =============================================================================


def _make_rubric_ground_truth(query: str, rubrics: list[dict]) -> str:
    """Build a JSON ground-truth string in the format expected by the rubric pipeline."""
    return json.dumps({"query": query, "rubrics": rubrics})


def _make_batch_ground_truths(queries_and_rubrics: list[tuple[str, list[dict]]], num_samples: int) -> list[list[str]]:
    """Build batch ground truths: each entry is ``[json_str]`` repeated ``num_samples`` times per query."""
    gts: list[list[str]] = []
    for query, rubrics in queries_and_rubrics:
        gt = [_make_rubric_ground_truth(query, rubrics)]
        for _ in range(num_samples):
            gts.append(gt)
    return gts


_STEP_SAMPLE_QUERIES = [
    ("What is the capital of France?", [{"description": "Correctly identifies Paris", "weight": 1.0}]),
    ("Explain photosynthesis.", [{"description": "Mentions sunlight", "weight": 1.0}]),
]

_FAKE_EVOLVING_RUBRICS = {
    "positive_rubrics": [{"description": "Provides historical context", "title": "HistContext"}],
    "negative_rubrics": [{"description": "Contains factual errors", "title": "FactErr"}],
}


class TestInitRubricBufferStep(unittest.TestCase):
    """Test ``init_rubric_buffer`` with rubric-formatted ground truths."""

    def test_basic_init(self):
        """Test buffer initialises persistent rubrics from ground truths."""
        gts = [_make_rubric_ground_truth(q, r) for q, r in _STEP_SAMPLE_QUERIES]
        buf = init_rubric_buffer(gts)

        self.assertEqual(len(buf), 2)
        for query, rubrics in _STEP_SAMPLE_QUERIES:
            self.assertIn(query, buf)
            entry = buf[query]
            self.assertEqual(len(entry["persistent_rubrics"]), len(rubrics))
            self.assertEqual(len(entry["active_rubrics"]), 0)
            self.assertEqual(len(entry["inactive_rubrics"]), 0)

    def test_wrapped_in_list(self):
        """Ground truths from the dataset come wrapped in a list."""
        gts = [[_make_rubric_ground_truth(q, r)] for q, r in _STEP_SAMPLE_QUERIES]
        buf = init_rubric_buffer(gts)
        self.assertEqual(len(buf), 2)

    def test_dedup_same_query(self):
        """Duplicate queries should produce a single buffer entry."""
        gts = [_make_rubric_ground_truth("Q1", [{"description": "d", "weight": 1.0}])] * 5
        buf = init_rubric_buffer(gts)
        self.assertEqual(len(buf), 1)


class TestEvolvingRubricConfig(unittest.TestCase):
    """Test ``EvolvingRubricConfig.from_streaming_config``."""

    def test_from_streaming_config(self):
        """Test config extraction from StreamingDataLoaderConfig."""
        mock_cfg = MagicMock()
        mock_cfg.apply_evolving_rubric_reward = True
        mock_cfg.num_samples_per_prompt_rollout = 8
        mock_cfg.max_active_rubrics = 3
        mock_cfg.cache_evolving_rubric_data_dir = "/tmp/cache"

        cfg = EvolvingRubricConfig.from_streaming_config(mock_cfg)
        self.assertTrue(cfg.apply_evolving_rubric_reward)
        self.assertEqual(cfg.num_samples_per_prompt_rollout, 8)
        self.assertEqual(cfg.max_active_rubrics, 3)
        self.assertEqual(cfg.cache_evolving_rubric_data_dir, "/tmp/cache")


class TestPushGroundTruthOverrides(unittest.TestCase):
    """Test ``_push_ground_truth_overrides`` without Ray."""

    def test_no_engines_noop(self):
        """No engines → no crash, no-op."""
        _push_ground_truth_overrides(["gt1", "gt2"], [0, 1], 1, [])

    def test_no_indices_noop(self):
        """No indices → engine method never called."""
        engine = MagicMock()
        _push_ground_truth_overrides(["gt1"], None, 1, [engine])
        engine.update_ground_truths.remote.assert_not_called()

    def test_deduplicates_indices(self):
        """Indices repeat for num_samples_per_prompt_rollout > 1."""
        overrides_seen: dict[int, Any] = {}

        class FakeEngine:
            class _Remote:
                @staticmethod
                def remote(overrides):
                    overrides_seen.update(overrides)
                    return MagicMock()

            update_ground_truths = _Remote()

        with patch("open_instruct.rubrics.evolving_rubric_step.ray") as mock_ray:
            mock_ray.get = lambda futures: None
            _push_ground_truth_overrides(
                updated_ground_truths=["gt_a", "gt_a", "gt_b", "gt_b"],
                indices=[0, 0, 1, 1],
                num_samples_per_prompt_rollout=2,
                vllm_engines=[FakeEngine()],
            )

        self.assertEqual(len(overrides_seen), 2)
        self.assertEqual(overrides_seen[0], "gt_a")
        self.assertEqual(overrides_seen[1], "gt_b")


class TestRunEvolvingRubricStepMocked(unittest.TestCase):
    """E2E test of ``run_evolving_rubric_step`` with mocked LLM calls."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 2
        self.ground_truths = _make_batch_ground_truths(_STEP_SAMPLE_QUERIES, self.num_samples)
        self.decoded_responses = [
            "Paris is the capital of France.",
            "The capital is Paris, a beautiful city.",
            "Photosynthesis converts sunlight into energy.",
            "Plants use sunlight via photosynthesis.",
        ]
        self.indices = [0, 0, 1, 1]
        self.config = EvolvingRubricConfig(
            apply_evolving_rubric_reward=True,
            num_samples_per_prompt_rollout=self.num_samples,
            max_active_rubrics=5,
        )
        gts_for_buffer = [_make_rubric_ground_truth(q, r) for q, r in _STEP_SAMPLE_QUERIES]
        self.rubric_buffer = init_rubric_buffer(gts_for_buffer)

    @patch("open_instruct.rubrics.rubric_utils.generate_instance_wise_evolving_rubrics")
    def test_step_returns_metrics_and_updated_buffer(self, mock_generate):
        """Test that a step returns expected metrics and updates the buffer."""
        mock_generate.return_value = _FAKE_EVOLVING_RUBRICS

        metrics, updated_buffer = run_evolving_rubric_step(
            decoded_responses=self.decoded_responses,
            ground_truths=self.ground_truths,
            indices=self.indices,
            config=self.config,
            rubric_buffer=self.rubric_buffer,
            vllm_engines=[],
            step=0,
        )

        self.assertIn("evolving_rubrics/valid_rate", metrics)
        self.assertIn("evolving_rubrics/num_new_rubrics", metrics)
        self.assertIn("evolving_rubrics/num_active_rubrics", metrics)
        self.assertGreater(metrics["evolving_rubrics/valid_rate"], 0)

        self.assertIsNotNone(updated_buffer)
        for query, _ in _STEP_SAMPLE_QUERIES:
            self.assertGreater(len(updated_buffer[query]["active_rubrics"]), 0)

    @patch("open_instruct.rubrics.rubric_utils.generate_instance_wise_evolving_rubrics")
    def test_step_with_cache_dir(self, mock_generate):
        """Test that cache files are written when a cache dir is specified."""
        mock_generate.return_value = _FAKE_EVOLVING_RUBRICS
        cache_dir = tempfile.mkdtemp()
        try:
            config = EvolvingRubricConfig(
                apply_evolving_rubric_reward=True,
                num_samples_per_prompt_rollout=self.num_samples,
                max_active_rubrics=5,
                cache_evolving_rubric_data_dir=cache_dir,
            )
            run_evolving_rubric_step(
                decoded_responses=self.decoded_responses,
                ground_truths=self.ground_truths,
                indices=self.indices,
                config=config,
                rubric_buffer=self.rubric_buffer,
                vllm_engines=[],
                step=42,
            )
            cache_files = os.listdir(cache_dir)
            self.assertGreater(len(cache_files), 0)
            self.assertTrue(any("step42" in f for f in cache_files))
        finally:
            shutil.rmtree(cache_dir)

    @patch("open_instruct.rubrics.rubric_utils.generate_instance_wise_evolving_rubrics")
    def test_step_with_none_rubrics(self, mock_generate):
        """LLM generation can fail and return None for some prompts."""
        mock_generate.side_effect = [None, _FAKE_EVOLVING_RUBRICS]

        metrics, updated_buffer = run_evolving_rubric_step(
            decoded_responses=self.decoded_responses,
            ground_truths=self.ground_truths,
            indices=self.indices,
            config=self.config,
            rubric_buffer=self.rubric_buffer,
            vllm_engines=[],
            step=0,
        )

        self.assertIn("evolving_rubrics/skipped", metrics)
        self.assertGreater(metrics["evolving_rubrics/skipped"], 0)

    @patch("open_instruct.rubrics.rubric_utils.generate_instance_wise_evolving_rubrics")
    def test_buffer_grows_across_steps(self, mock_generate):
        """Active rubrics should accumulate across multiple steps."""
        mock_generate.return_value = _FAKE_EVOLVING_RUBRICS

        _, buf1 = run_evolving_rubric_step(
            decoded_responses=self.decoded_responses,
            ground_truths=self.ground_truths,
            indices=self.indices,
            config=self.config,
            rubric_buffer=self.rubric_buffer,
            vllm_engines=[],
            step=0,
        )
        active_count_1 = len(buf1[_STEP_SAMPLE_QUERIES[0][0]]["active_rubrics"])

        mock_generate.return_value = {
            "positive_rubrics": [{"description": "New rubric", "title": "New"}],
            "negative_rubrics": [],
        }
        _, buf2 = run_evolving_rubric_step(
            decoded_responses=self.decoded_responses,
            ground_truths=self.ground_truths,
            indices=self.indices,
            config=self.config,
            rubric_buffer=buf1,
            vllm_engines=[],
            step=1,
        )
        active_count_2 = len(buf2[_STEP_SAMPLE_QUERIES[0][0]]["active_rubrics"])

        self.assertGreater(active_count_2, active_count_1)


class TestBatchIndicesFromModel(unittest.TestCase):
    """Verify that ``Batch.indices`` is populated correctly by the index-collecting logic."""

    def test_indices_structure(self):
        """Batch.indices should repeat each dataset index num_samples_per_prompt times."""
        num_prompts = 3
        num_samples = 2
        queries = [[1, 2]] * (num_prompts * num_samples)
        ground_truths = [["gt"]] * (num_prompts * num_samples)
        datasets_list = ["ds"] * (num_prompts * num_samples)
        raw_queries = ["raw"] * (num_prompts * num_samples)
        decoded = ["resp"] * (num_prompts * num_samples)
        scores = [0.5] * (num_prompts * num_samples)

        indices = []
        for i in range(num_prompts):
            for _ in range(num_samples):
                indices.append(i)

        batch = Batch(
            queries=queries,
            ground_truths=ground_truths,
            datasets=datasets_list,
            raw_queries=raw_queries,
            decoded_responses=decoded,
            indices=indices,
            scores=scores,
        )

        self.assertIsNotNone(batch.indices)
        self.assertEqual(len(batch.indices), num_prompts * num_samples)
        self.assertEqual(batch.indices, [0, 0, 1, 1, 2, 2])

    def test_batch_slicing_preserves_indices(self):
        """Slicing a Batch should preserve the indices."""
        batch = Batch(
            queries=[[1]] * 4,
            ground_truths=[["gt"]] * 4,
            datasets=["ds"] * 4,
            raw_queries=["raw"] * 4,
            decoded_responses=["resp"] * 4,
            indices=[10, 10, 20, 20],
            scores=[0.5] * 4,
        )

        sliced = batch[2:]
        self.assertEqual(sliced.indices, [20, 20])


@unittest.skipIf(not _check_litellm_available(), "litellm not installed")
@unittest.skipIf(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_API_KEY")),
    "No API credentials available",
)
class TestRunEvolvingRubricStepWithAPI(unittest.TestCase):
    """Full E2E test with real LLM calls (requires API)."""

    def test_live_rubric_generation(self):
        """Test that a single evolving rubric step generates new rubrics."""
        os.environ.setdefault("RUBRIC_GENERATION_MODEL", "gpt-4.1-mini")

        num_samples = 2
        ground_truths = _make_batch_ground_truths(_STEP_SAMPLE_QUERIES, num_samples)
        decoded_responses = [
            "Paris is the capital of France, known for the Eiffel Tower.",
            "The capital of France is Paris.",
            "Photosynthesis converts sunlight to chemical energy in plants.",
            "Plants absorb sunlight for photosynthesis.",
        ]
        indices = [0, 0, 1, 1]

        gts_for_buffer = [_make_rubric_ground_truth(q, r) for q, r in _STEP_SAMPLE_QUERIES]
        rubric_buffer = init_rubric_buffer(gts_for_buffer)

        config = EvolvingRubricConfig(
            apply_evolving_rubric_reward=True,
            num_samples_per_prompt_rollout=num_samples,
            max_active_rubrics=5,
        )

        metrics, updated_buffer = run_evolving_rubric_step(
            decoded_responses=decoded_responses,
            ground_truths=ground_truths,
            indices=indices,
            config=config,
            rubric_buffer=rubric_buffer,
            vllm_engines=[],
            step=0,
        )

        self.assertIn("evolving_rubrics/valid_rate", metrics)
        self.assertIn("evolving_rubrics/num_new_rubrics", metrics)
        self.assertGreater(metrics["evolving_rubrics/valid_rate"], 0, "At least one rubric should be generated")

        self.assertIsNotNone(updated_buffer)
        total_active = sum(len(v["active_rubrics"]) for v in updated_buffer.values())
        self.assertGreater(total_active, 0, "Buffer should have active rubrics after generation")

    def test_live_two_step_buffer_growth(self):
        """Run two steps to verify buffer accumulates across steps."""
        os.environ.setdefault("RUBRIC_GENERATION_MODEL", "gpt-4.1-mini")

        num_samples = 2
        ground_truths = _make_batch_ground_truths(_STEP_SAMPLE_QUERIES, num_samples)
        decoded_responses = [
            "Paris is the capital.",
            "France's capital is Paris.",
            "Photosynthesis uses sunlight.",
            "Plants convert sunlight.",
        ]
        indices = [0, 0, 1, 1]

        gts_for_buffer = [_make_rubric_ground_truth(q, r) for q, r in _STEP_SAMPLE_QUERIES]
        rubric_buffer = init_rubric_buffer(gts_for_buffer)

        config = EvolvingRubricConfig(
            apply_evolving_rubric_reward=True,
            num_samples_per_prompt_rollout=num_samples,
            max_active_rubrics=10,
        )

        _, buf_step0 = run_evolving_rubric_step(
            decoded_responses=decoded_responses,
            ground_truths=ground_truths,
            indices=indices,
            config=config,
            rubric_buffer=rubric_buffer,
            vllm_engines=[],
            step=0,
        )

        active_step0 = sum(len(v["active_rubrics"]) for v in buf_step0.values())

        _, buf_step1 = run_evolving_rubric_step(
            decoded_responses=decoded_responses,
            ground_truths=ground_truths,
            indices=indices,
            config=config,
            rubric_buffer=buf_step0,
            vllm_engines=[],
            step=1,
        )

        active_step1 = sum(len(v["active_rubrics"]) for v in buf_step1.values())
        self.assertGreaterEqual(active_step1, active_step0)


if __name__ == "__main__":
    unittest.main()
