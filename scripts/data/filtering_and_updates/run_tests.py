#!/usr/bin/env python3
"""
Test runner for filtering scripts
Runs all available tests in the filtering_and_updates directory.
"""

import glob
import os
import subprocess
import sys


def run_test_file(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*80}")
    print(f"🧪 RUNNING TEST: {test_file}")
    print(f"{'='*80}")

    try:
        result = subprocess.run([sys.executable, test_file], capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode == 0:
            print("✅ TEST PASSED")
            print(result.stdout)
            return True
        else:
            print("❌ TEST FAILED")
            print(result.stdout)
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ ERROR RUNNING TEST: {e}")
        return False


def main():
    """Run all test files in the directory."""
    print("🧪 FILTERING SCRIPTS TEST RUNNER")
    print("=" * 80)

    # Find all test files
    test_files = glob.glob("test_*.py")

    if not test_files:
        print("No test files found!")
        return

    print(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file}")

    # Run all tests
    passed = 0
    failed = 0

    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
