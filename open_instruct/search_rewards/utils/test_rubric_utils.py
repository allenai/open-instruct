#!/usr/bin/env python3
"""Test script for rubric_utils.py - tests adaptive rubric generation functions.

Usage:
    # From the repo root:
    python -m open_instruct.search_rewards.utils.test_rubric_utils

    # Or directly:
    python open_instruct/search_rewards/utils/test_rubric_utils.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile

# Ensure the repo root is in the Python path
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from open_instruct.search_rewards.utils.rubric_utils import (  # noqa: E402
    initialize_rubric_buffer,
    save_adaptive_rubric_cache_safe,
    update_ground_truths_with_adaptive_rubrics,
)


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def test_initialize_rubric_buffer():
    """Test rubric buffer initialization."""
    print("=" * 60)
    print("Testing initialize_rubric_buffer()")
    print("=" * 60)

    # Create mock ground truths list (the function takes ground_truths directly)
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

    # Test with static rubrics as persistent
    buffer = initialize_rubric_buffer(ground_truths, use_static_rubrics_as_persistent=True)

    # Check buffer structure
    assert len(buffer) == 2, f"Expected 2 entries, got {len(buffer)}"
    assert "What is the capital of France?" in buffer
    assert "Explain photosynthesis." in buffer

    # Check rubric structure for first entry
    france_entry = buffer["What is the capital of France?"]
    assert "active_rubrics" in france_entry
    assert "inactive_rubrics" in france_entry
    assert "persistent_rubrics" in france_entry

    # With use_static_rubrics_as_persistent=True, active should be empty
    assert len(france_entry["active_rubrics"]) == 0, "Active rubrics should be empty"
    assert len(france_entry["persistent_rubrics"]) == 2, "Persistent rubrics should have 2 items"

    print("  ✅ Test 1: Buffer structure is correct")

    # Test with static rubrics as active (not persistent)
    buffer2 = initialize_rubric_buffer(ground_truths, use_static_rubrics_as_persistent=False)

    france_entry2 = buffer2["What is the capital of France?"]
    assert len(france_entry2["active_rubrics"]) == 2, "Active rubrics should have 2 items"
    assert len(france_entry2["persistent_rubrics"]) == 0, "Persistent rubrics should be empty"

    print("  ✅ Test 2: Static as active rubrics works")
    print("\n✅ All initialize_rubric_buffer tests passed!")
    return True


def test_update_ground_truths_with_adaptive_rubrics():
    """Test updating ground truths with adaptive rubrics."""
    print("=" * 60)
    print("Testing update_ground_truths_with_adaptive_rubrics()")
    print("=" * 60)

    # Create mock ground truths (2 prompts, 2 samples each = 4 total)
    ground_truths = [
        [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
        [json.dumps({"query": "Q1", "rubrics": [{"description": "R1", "weight": 1.0}]})],
        [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
        [json.dumps({"query": "Q2", "rubrics": [{"description": "R2", "weight": 1.0}]})],
    ]

    # Mock adaptive rubrics (one per unique prompt)
    all_adaptive_rubrics = [
        {
            "question": "Q1",
            "positive_rubrics": [{"description": "Positive1", "title": "P1"}],
            "negative_rubrics": [{"description": "Negative1", "title": "N1"}],
        },
        {"question": "Q2", "positive_rubrics": [{"description": "Positive2", "title": "P2"}], "negative_rubrics": []},
    ]

    num_samples_per_prompt_rollout = 2

    # Test without rubric buffer
    updated_gts, rate, avg_gt, avg_ar, avg_active_buf, buffer, skipped = update_ground_truths_with_adaptive_rubrics(
        ground_truths.copy(), all_adaptive_rubrics, num_samples_per_prompt_rollout, rubric_buffer=None
    )

    # Verify results
    assert len(updated_gts) == 4, f"Expected 4 ground truths, got {len(updated_gts)}"

    # Check first prompt's ground truths
    gt1 = json.loads(updated_gts[0][0])
    assert len(gt1["rubrics"]) == 3, f"Expected 3 rubrics (1 original + 2 adaptive), got {len(gt1['rubrics'])}"
    assert gt1["rubrics"][0]["description"] == "R1"  # Original
    assert gt1["rubrics"][1]["description"] == "Positive1"  # Positive adaptive
    assert gt1["rubrics"][1]["weight"] == 1.0
    assert gt1["rubrics"][2]["description"] == "Negative1"  # Negative adaptive
    assert gt1["rubrics"][2]["weight"] == -1.0  # Negative weight

    print("  ✅ Test 1: Adaptive rubrics added correctly")

    # Check second prompt's ground truths
    gt3 = json.loads(updated_gts[2][0])
    assert len(gt3["rubrics"]) == 2, f"Expected 2 rubrics, got {len(gt3['rubrics'])}"

    print("  ✅ Test 2: Second prompt rubrics correct")

    # Test with None (skipped) adaptive rubrics
    adaptive_with_none = [None, all_adaptive_rubrics[1]]
    updated_gts2, rate2, _, _, _, _, skipped2 = update_ground_truths_with_adaptive_rubrics(
        ground_truths.copy(), adaptive_with_none, num_samples_per_prompt_rollout, rubric_buffer=None
    )

    assert skipped2 == 2, f"Expected 2 skipped, got {skipped2}"  # None affects 2 samples
    print("  ✅ Test 3: Handles None rubrics correctly")

    print("\n✅ All update_ground_truths tests passed!")
    return True


def test_save_adaptive_rubric_cache_safe():
    """Test safe caching of adaptive rubric data."""
    print("=" * 60)
    print("Testing save_adaptive_rubric_cache_safe()")
    print("=" * 60)

    # Create temp directory
    cache_dir = tempfile.mkdtemp()

    test_data = {
        "decoded_responses": ["Response 1", "Response 2"],
        "ground_truths": [json.dumps({"query": "Q1", "rubrics": []}), json.dumps({"query": "Q2", "rubrics": []})],
        "all_adaptive_rubrics": [
            {"positive_rubrics": [], "negative_rubrics": []},
            {"positive_rubrics": [], "negative_rubrics": []},
        ],
        "num_subsampled_answers_list": [2, 2],
        "num_samples_per_prompt_rollout": 2,
        "use_full_responses": True,
        "answer_length_limit_in_words": None,
    }

    # Save cache
    cache_path = save_adaptive_rubric_cache_safe(cache_dir=cache_dir, training_step=1, **test_data)

    # Verify file was created
    assert os.path.exists(cache_path), f"Cache file not created: {cache_path}"
    print(f"  ✅ Test 1: Cache file created at {cache_path}")

    # Verify file contents
    with open(cache_path) as f:
        loaded_data = json.load(f)

    assert loaded_data["training_step"] == 1
    assert loaded_data["inputs"]["decoded_responses"] == test_data["decoded_responses"]
    assert loaded_data["outputs"]["all_adaptive_rubrics"] == test_data["all_adaptive_rubrics"]
    print("  ✅ Test 2: Cache contents are correct")

    # Test multiple saves don't conflict
    cache_path2 = save_adaptive_rubric_cache_safe(cache_dir=cache_dir, training_step=2, **test_data)
    assert os.path.exists(cache_path2)
    assert cache_path != cache_path2
    print("  ✅ Test 3: Multiple saves create unique files")

    # Cleanup
    shutil.rmtree(cache_dir)
    print("\n✅ All save_adaptive_rubric_cache_safe tests passed!")
    return True


async def test_generate_instance_wise_adaptive_rubrics():
    """Test adaptive rubric generation with real API calls."""
    print("=" * 60)
    print("Testing generate_instance_wise_adaptive_rubrics()")
    print("=" * 60)

    if not _check_litellm_available():
        print("⚠️  Skipping - litellm is not installed")
        print("   Install with: pip install litellm")
        return True

    # Check if we have API credentials
    has_openai = os.environ.get("OPENAI_API_KEY") is not None
    has_azure = os.environ.get("AZURE_API_KEY") is not None

    if not has_openai and not has_azure:
        print("⚠️  Skipping API test - no credentials found")
        print("   Set OPENAI_API_KEY or AZURE_API_KEY to run this test")
        return True

    # Import here since we've verified litellm is available
    from open_instruct.search_rewards.utils.rubric_utils import (  # noqa: PLC0415
        generate_instance_wise_adaptive_rubrics,
    )

    # Test data
    question = "What is the capital of France?"
    response_list = [
        "The capital of France is Paris. Paris is known for the Eiffel Tower.",
        "Paris is the capital.",
        "France's capital city is Paris, which has been the capital since the 10th century.",
    ]
    existing_rubrics = json.dumps([{"description": "Correctly identifies Paris as the capital", "weight": 1.0}])

    model = os.environ.get("RUBRIC_GENERATION_MODEL", "gpt-4.1-mini")
    print(f"  Using model: {model}")

    # Call the function
    result = await generate_instance_wise_adaptive_rubrics(
        question=question, response_list=response_list, existing_rubrics=existing_rubrics, model_name=model
    )

    print(f"  Result: {json.dumps(result, indent=2)}")

    # Verify result structure
    assert result is not None, "Result should not be None"
    assert "positive_rubrics" in result or "negative_rubrics" in result, "Should have rubric lists"

    print("  ✅ Adaptive rubric generation succeeded!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUBRIC_UTILS.PY TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Tests that don't require API or litellm
    results.append(("initialize_rubric_buffer", test_initialize_rubric_buffer()))
    results.append(("update_ground_truths_with_adaptive_rubrics", test_update_ground_truths_with_adaptive_rubrics()))
    results.append(("save_adaptive_rubric_cache_safe", test_save_adaptive_rubric_cache_safe()))

    # Tests that require litellm + API
    results.append(
        ("generate_instance_wise_adaptive_rubrics", asyncio.run(test_generate_instance_wise_adaptive_rubrics()))
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
