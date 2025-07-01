"""
Integration test for the /test_program endpoint.

Put this file alongside your source, adjust nothing else,
and run `python -m unittest -v` (or any test runner you like).
"""

import logging
import os
import subprocess
import time
import unittest

import requests

# Set up global logger
logger = logging.getLogger(__name__)


class APITestServer:
    """Manages starting and stopping the API server for testing."""

    def __init__(self, host="0.0.0.0", port=1234, startup_timeout=30):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.base_url = f"http://localhost:{port}"
        self.health_url = f"{self.base_url}/health"
        self.process = None

    def is_running(self):
        """Check if the server is already running."""
        try:
            response = requests.get(self.health_url, timeout=1)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start(self):
        """Start the server if it's not already running."""
        if self.is_running():
            logger.info("Server already running, using existing instance")
            return True

        logger.info("Starting API server...")
        self.process = subprocess.Popen(
            ["uv", "run", "uvicorn", "open_instruct.code.api:app", "--host", self.host, "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        for _ in range(self.startup_timeout):
            if self.is_running():
                logger.info("Server started successfully")
                return True
            time.sleep(1)

        # Server failed to start
        if self.process:
            self.process.terminate()
            self.process = None
        raise RuntimeError(f"Failed to start server within {self.startup_timeout} seconds")

    def stop(self):
        """Stop the server if we started it."""
        if self.process:
            logger.info("Stopping API server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.stop()
        return False


class TestAddProgramAPI(unittest.TestCase):
    """Replicates the manual `requests.post` call with real assertions."""

    # ---- class-level fixtures -------------------------------------------------
    @classmethod
    def setUpClass(cls):
        # Initialize the test server
        cls.server = APITestServer()
        cls.server.start()

        # Allow overriding the URL via env var for CI flexibility.
        cls.url = os.getenv("ADD_API_URL", f"{cls.server.base_url}/test_program")

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

    @classmethod
    def tearDownClass(cls):
        # Stop the server
        cls.server.stop()

    # ---- actual test ----------------------------------------------------------
    def test_add_program_results(self):
        """POST to the endpoint and verify JSON response structure & content."""
        try:
            response = requests.post(self.url, json=self.payload, timeout=10)
        except requests.exceptions.RequestException as exc:
            self.fail(f"Failed to connect to {self.url}: {exc}")

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


class TestAPIServerContextManager(unittest.TestCase):
    """Example of using the TestAPIServer as a context manager."""

    def test_health_check_with_context_manager(self):
        """Test using the server context manager for a single test."""
        with APITestServer() as server:
            response = requests.get(server.health_url, timeout=5)
            self.assertEqual(response.status_code, 200)

            # Test that the health endpoint returns expected structure
            data = response.json()
            self.assertIn("status", data)


class TestAPIServerStandalone(unittest.TestCase):
    """Example of using the TestAPIServer manually in individual tests."""

    def setUp(self):
        self.server = APITestServer()

    def tearDown(self):
        self.server.stop()

    def test_health_endpoint(self):
        """Test the health endpoint."""
        self.server.start()
        response = requests.get(self.server.health_url, timeout=5)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
