"""
Integration test for the /test_program endpoint.

Put this file alongside your source, adjust nothing else,
and run `python -m unittest -v` (or any test runner you like).
"""

import os
import unittest
import requests


class TestAddProgramAPI(unittest.TestCase):
    """Replicates the manual `requests.post` call with real assertions."""

    # ---- class-level fixtures -------------------------------------------------
    @classmethod
    def setUpClass(cls):
        # Allow overriding the URL via env var for CI flexibility.
        cls.url = os.getenv("ADD_API_URL", "http://localhost:1234/test_program")

        cls.payload = {
            "program": ("def add(a, b):\n    return a + b\n"),
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(0, 0) == 1",  # expected to fail
            ],
            "max_execution_time": 1.0,
        }

        # Expected result mirrors the original script: first two pass, last fails.
        cls.expected_results = [1, 1, 0]

    # ---- actual test ----------------------------------------------------------
    def test_add_program_results(self):
        """POST to the endpoint and verify JSON response structure & content."""
        try:
            response = requests.post(self.url, json=self.payload, timeout=5)
        except requests.exceptions.ConnectionError as exc:
            self.skipTest(f"Could not reach {self.url!r}: {exc}")

        self.assertEqual(
            response.status_code,
            200,
            f"Unexpected status code from {self.url}: {response.status_code}",
        )
            
        data = response.json()
        self.assertIn("results", data, "Response JSON missing 'results' field")

        self.assertEqual(
            data["results"],
            self.expected_results,
            "Returned pass/fail vector does not match expectation",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
