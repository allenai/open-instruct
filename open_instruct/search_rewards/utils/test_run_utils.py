#!/usr/bin/env python3
"""Test script for run_utils.py - tests LiteLLM calls and JSON extraction.

Usage:
    # From the repo root:
    python -m open_instruct.search_rewards.utils.test_run_utils

    # Or directly:
    python open_instruct/search_rewards/utils/test_run_utils.py
"""

import asyncio
import os
import sys

# Ensure the repo root is in the Python path
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import only the JSON extraction function at module level (doesn't need litellm)
from open_instruct.search_rewards.utils.run_utils import extract_json_from_response  # noqa: E402


def test_extract_json_from_response():
    """Test JSON extraction from various response formats."""
    print("=" * 60)
    print("Testing extract_json_from_response()")
    print("=" * 60)

    test_cases = [
        # Basic JSON
        ('{"score": 2}', {"score": 2}),
        # JSON with text before
        ('The score is {"score": 1}', {"score": 1}),
        # JSON with text after
        ('{"score": 0} is the result', {"score": 0}),
        # JSON in markdown code block
        ('```json\n{"score": 2}\n```', {"score": 2}),
        # Nested JSON
        ('{"score": 1, "reason": "good"}', {"score": 1, "reason": "good"}),
        # Double braces (common LLM mistake)
        ('{{"score": 2}}', {"score": 2}),
        # No JSON
        ("This has no JSON", None),
        # Invalid JSON
        ('{"score": }', None),
        # Multiple JSON objects (should get last valid one)
        ('{"a": 1} more text {"score": 2}', {"score": 2}),
    ]

    passed = 0
    failed = 0
    for i, (input_str, expected) in enumerate(test_cases):
        result = extract_json_from_response(input_str)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  Test {i + 1}: {status}")
        print(f"    Input: {input_str[:50]}...")
        print(f"    Expected: {expected}")
        print(f"    Got: {result}")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def _check_litellm_available():
    """Check if litellm is available."""
    try:
        import litellm  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


async def test_run_litellm_async():
    """Test async LiteLLM calls (requires API key)."""
    print("=" * 60)
    print("Testing run_litellm_async()")
    print("=" * 60)

    if not _check_litellm_available():
        print("⚠️  Skipping - litellm is not installed")
        print("   Install with: pip install litellm")
        return True

    # Check if we have API credentials
    has_openai = os.environ.get("OPENAI_API_KEY") is not None
    has_azure = os.environ.get("AZURE_API_KEY") is not None

    if not has_openai and not has_azure:
        print("⚠️  Skipping LiteLLM test - no API credentials found")
        print("   Set OPENAI_API_KEY or AZURE_API_KEY to run this test")
        return True

    # Import here since we've verified litellm is available
    from open_instruct.search_rewards.utils.run_utils import run_litellm_async  # noqa: PLC0415

    model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")
    print(f"  Using model: {model}")

    response = await run_litellm_async(
        model_name=model, user_prompt="What is 2+2? Answer with just the number.", temperature=0, max_tokens=10
    )
    print(f"  Response: {response}")
    assert response.strip() in ["4", "4."], f"Expected '4', got '{response}'"
    print("  ✅ PASS - LiteLLM async call succeeded")
    return True


def test_run_litellm_sync():
    """Test sync LiteLLM calls (requires API key)."""
    print("=" * 60)
    print("Testing run_litellm() [sync]")
    print("=" * 60)

    if not _check_litellm_available():
        print("⚠️  Skipping - litellm is not installed")
        return True

    # Check if we have API credentials
    has_openai = os.environ.get("OPENAI_API_KEY") is not None
    has_azure = os.environ.get("AZURE_API_KEY") is not None

    if not has_openai and not has_azure:
        print("⚠️  Skipping LiteLLM test - no API credentials found")
        return True

    # Import here since we've verified litellm is available
    from open_instruct.search_rewards.utils.run_utils import run_litellm  # noqa: PLC0415

    model = os.environ.get("RUBRIC_JUDGE_MODEL", "gpt-4.1-mini")
    print(f"  Using model: {model}")

    response = run_litellm(
        model_name=model, user_prompt="What is 3+3? Answer with just the number.", temperature=0, max_tokens=10
    )
    print(f"  Response: {response}")
    assert response.strip() in ["6", "6."], f"Expected '6', got '{response}'"
    print("  ✅ PASS - LiteLLM sync call succeeded")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUN_UTILS.PY TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Test JSON extraction (no API needed, no litellm needed)
    results.append(("extract_json_from_response", test_extract_json_from_response()))

    # Test sync LiteLLM (requires litellm + API key)
    results.append(("run_litellm", test_run_litellm_sync()))

    # Test async LiteLLM (requires litellm + API key)
    results.append(("run_litellm_async", asyncio.run(test_run_litellm_async())))

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
