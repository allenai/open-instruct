#!/usr/bin/env python3
"""Master test script that runs all adaptive rubric tests.

Usage:
    # From the repo root:
    python -m open_instruct.search_rewards.utils.test_all

    # Or directly:
    python open_instruct/search_rewards/utils/test_all.py
"""

import os
import subprocess
import sys

# Get the directory containing this script and repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))


def run_test_file(test_file: str) -> bool:
    """Run a test file and return whether it passed."""
    print(f"\n{'=' * 60}")
    print(f"Running: {os.path.basename(test_file)}")
    print("=" * 60)

    # Run the test file directly, setting PYTHONPATH to include repo root
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run([sys.executable, test_file], env=env, capture_output=False)
    return result.returncode == 0


def main():
    """Run all test files."""
    print("\n" + "=" * 60)
    print("ADAPTIVE RUBRICS - FULL TEST SUITE")
    print("=" * 60)

    test_files = [
        os.path.join(_SCRIPT_DIR, "test_run_utils.py"),
        os.path.join(_SCRIPT_DIR, "test_rubric_utils.py"),
        os.path.join(_SCRIPT_DIR, "test_rubric_verifier.py"),
    ]

    results = []
    for test_file in test_files:
        passed = run_test_file(test_file)
        results.append((os.path.basename(test_file), passed))

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'✅ ALL TEST SUITES PASSED' if all_passed else '❌ SOME TEST SUITES FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
