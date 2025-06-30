import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from open_instruct.code.api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    @patch("open_instruct.code.api.get_successful_tests_fast")
    def test_test_program_success(self, mock_get_successful_tests):
        """Test successful program testing"""
        # Mock the function to return expected results
        mock_get_successful_tests.return_value = [1, 1, 0]

        payload = {
            "program": "def add(a, b):\n    return a + b",
            "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"],
            "max_execution_time": 1.0,
        }

        response = self.client.post("/test_program", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"results": [1, 1, 0]})

        # Verify the mock was called with correct arguments
        mock_get_successful_tests.assert_called_once_with(
            program=payload["program"], tests=payload["tests"], max_execution_time=payload["max_execution_time"]
        )

    @patch("open_instruct.code.api.get_successful_tests_fast")
    def test_test_program_default_timeout(self, mock_get_successful_tests):
        """Test program testing with default timeout"""
        mock_get_successful_tests.return_value = [1, 1]

        payload = {
            "program": "def add(a, b):\n    return a + b",
            "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        }

        response = self.client.post("/test_program", json=payload)

        self.assertEqual(response.status_code, 200)
        # Verify default timeout of 1.0 was used
        mock_get_successful_tests.assert_called_once_with(
            program=payload["program"], tests=payload["tests"], max_execution_time=1.0
        )

    @patch("open_instruct.code.api.get_successful_tests_fast")
    def test_test_program_exception(self, mock_get_successful_tests):
        """Test error handling when get_successful_tests_fast raises exception"""
        mock_get_successful_tests.side_effect = Exception("Test error")

        payload = {
            "program": "def add(a, b):\n    return a + b",
            "tests": ["assert add(1, 2) == 3"],
        }

        response = self.client.post("/test_program", json=payload)

        self.assertEqual(response.status_code, 500)
        self.assertIn("Test error", response.json()["detail"])

    def test_test_program_invalid_json(self):
        """Test with invalid JSON payload"""
        response = self.client.post("/test_program", json={})

        # Should get validation error for missing required fields
        self.assertEqual(response.status_code, 422)

    def test_test_program_invalid_types(self):
        """Test with invalid data types"""
        payload = {
            "program": 123,  # Should be string
            "tests": "not a list",  # Should be list
            "max_execution_time": "not a float",  # Should be float
        }

        response = self.client.post("/test_program", json=payload)
        self.assertEqual(response.status_code, 422)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests that would normally run against a live server"""

    def setUp(self):
        self.client = TestClient(app)

    def test_add_function_integration(self):
        """Integration test with actual add function"""
        payload = {
            "program": """
def add(a, b):
    return a + b
""",
            "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"],
            "max_execution_time": 1.0,
        }

        response = self.client.post("/test_program", json=payload)

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("results", response_data)
        # Third test should fail: add(0, 0) == 0, not 1
        self.assertEqual(response_data["results"], [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
