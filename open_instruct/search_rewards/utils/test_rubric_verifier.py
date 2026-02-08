#!/usr/bin/env python3
"""Test script for RubricVerifier in ground_truth_utils.py.

NOTE: This test requires the full project environment with numpy, etc.
Run with: python -m open_instruct.search_rewards.utils.test_rubric_verifier

If numpy is not installed, run: pip install numpy
"""

import asyncio
import json
import os
import sys

# Ensure the repo root is in the Python path
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _check_numpy_available():
    """Check if numpy is available."""
    try:
        import numpy  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def test_rubric_verifier_config():
    """Test RubricVerifierConfig creation."""
    print("=" * 60)
    print("Testing RubricVerifierConfig")
    print("=" * 60)

    if not _check_numpy_available():
        print("⚠️  Skipping - numpy is not installed")
        print("   Install with: pip install numpy")
        return True

    from open_instruct.ground_truth_utils import RubricVerifierConfig  # noqa: PLC0415

    # Test default config
    config = RubricVerifierConfig()
    assert config.rubric_judge_model == "gpt-4.1"
    assert config.rubric_judge_max_tokens == 2048
    assert config.rubric_judge_temperature == 0.0
    print("  ✅ Test 1: Default config values correct")

    # Test custom config
    config2 = RubricVerifierConfig(
        rubric_judge_model="gpt-4.1-mini", rubric_judge_max_tokens=1024, rubric_judge_temperature=0.5
    )
    assert config2.rubric_judge_model == "gpt-4.1-mini"
    assert config2.rubric_judge_max_tokens == 1024
    print("  ✅ Test 2: Custom config values correct")

    print("\n✅ All RubricVerifierConfig tests passed!")
    return True


def test_rubric_verifier_creation():
    """Test RubricVerifier instantiation."""
    print("=" * 60)
    print("Testing RubricVerifier creation")
    print("=" * 60)

    if not _check_numpy_available():
        print("⚠️  Skipping - numpy is not installed")
        return True

    from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig  # noqa: PLC0415

    config = RubricVerifierConfig()
    verifier = RubricVerifier(config)

    assert verifier.name == "rubric"
    assert verifier.weight == 1.0
    assert verifier.verifier_config == config
    print("  ✅ Test 1: Verifier created successfully")

    # Test get_config_class
    assert RubricVerifier.get_config_class() == RubricVerifierConfig
    print("  ✅ Test 2: get_config_class returns correct class")

    print("\n✅ All RubricVerifier creation tests passed!")
    return True


async def test_rubric_verifier_scoring():
    """Test RubricVerifier scoring with real API calls."""
    print("=" * 60)
    print("Testing RubricVerifier scoring")
    print("=" * 60)

    if not _check_numpy_available():
        print("⚠️  Skipping - numpy is not installed")
        return True

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

    from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig  # noqa: PLC0415

    config = RubricVerifierConfig(rubric_judge_model=os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini"))
    verifier = RubricVerifier(config)

    # Test case 1: Good response
    prediction_good = "The capital of France is Paris. Paris is a major European city."
    label_good = json.dumps(
        {
            "query": "What is the capital of France?",
            "rubrics": [{"description": "Correctly identifies Paris as the capital of France", "weight": 1.0}],
        }
    )

    result_good = await verifier.async_call(
        tokenized_prediction=[], prediction=prediction_good, label=label_good, query="What is the capital of France?"
    )

    print(f"  Good response score: {result_good.score}")
    assert result_good.score >= 0.5, f"Good response should score high, got {result_good.score}"
    print("  ✅ Test 1: Good response scores high")

    # Test case 2: Bad response
    prediction_bad = "The capital of France is London."
    result_bad = await verifier.async_call(
        tokenized_prediction=[], prediction=prediction_bad, label=label_good, query="What is the capital of France?"
    )

    print(f"  Bad response score: {result_bad.score}")
    assert result_bad.score < result_good.score, "Bad response should score lower than good"
    print("  ✅ Test 2: Bad response scores low")

    # Test case 3: Multiple rubrics with weights
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
    prediction_multi = "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen."

    result_multi = await verifier.async_call(
        tokenized_prediction=[],
        prediction=prediction_multi,
        label=label_multi,
        query="Explain photosynthesis briefly.",
    )

    print(f"  Multi-rubric score: {result_multi.score}")
    assert 0 <= result_multi.score <= 1, f"Score should be between 0 and 1, got {result_multi.score}"
    print("  ✅ Test 3: Multi-rubric scoring works")

    print("\n✅ All RubricVerifier scoring tests passed!")
    return True


async def test_rubric_verifier_edge_cases():
    """Test RubricVerifier edge cases."""
    print("=" * 60)
    print("Testing RubricVerifier edge cases")
    print("=" * 60)

    if not _check_numpy_available():
        print("⚠️  Skipping - numpy is not installed")
        return True

    from open_instruct.ground_truth_utils import RubricVerifier, RubricVerifierConfig  # noqa: PLC0415

    config = RubricVerifierConfig()
    verifier = RubricVerifier(config)

    # Test case 1: Empty rubrics
    label_empty = json.dumps({"query": "Test question", "rubrics": []})
    result_empty = await verifier.async_call(
        tokenized_prediction=[], prediction="Some response", label=label_empty, query="Test question"
    )
    assert result_empty.score == 0.0, "Empty rubrics should return 0"
    print("  ✅ Test 1: Empty rubrics handled correctly")

    # Test case 2: Invalid JSON label
    result_invalid = await verifier.async_call(
        tokenized_prediction=[], prediction="Some response", label="not valid json", query="Test question"
    )
    assert result_invalid.score == 0.0, "Invalid JSON should return 0"
    print("  ✅ Test 2: Invalid JSON handled correctly")

    # Test case 3: Label is not a dict
    result_list = await verifier.async_call(
        tokenized_prediction=[],
        prediction="Some response",
        label=json.dumps(["not", "a", "dict"]),
        query="Test question",
    )
    assert result_list.score == 0.0, "Non-dict label should return 0"
    print("  ✅ Test 3: Non-dict label handled correctly")

    print("\n✅ All RubricVerifier edge case tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUBRIC_VERIFIER TEST SUITE")
    print("=" * 60 + "\n")

    if not _check_numpy_available():
        print("⚠️  numpy is not installed - most tests will be skipped")
        print("   Install the project dependencies to run full tests")
        print()

    results = []

    # Tests that require numpy
    results.append(("RubricVerifierConfig", test_rubric_verifier_config()))
    results.append(("RubricVerifier creation", test_rubric_verifier_creation()))
    results.append(("RubricVerifier edge cases", asyncio.run(test_rubric_verifier_edge_cases())))

    # Tests that require numpy + litellm + API
    results.append(("RubricVerifier scoring", asyncio.run(test_rubric_verifier_scoring())))

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
