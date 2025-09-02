import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import run_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..utils.run_utils import extract_json_from_response, run_chatopenai


class TestRunUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.system_prompt = "You are a helpful assistant."
        self.user_prompt = "What is 2 + 2?"
        self.model_name = "gpt-3.5-turbo"

    @patch("run_utils.litellm.completion")
    def test_run_chatopenai_basic(self, mock_completion):
        """Test basic functionality of run_chatopenai"""
        # Mock the litellm completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "2 + 2 equals 4."
        mock_completion.return_value = mock_response

        # Test the function
        result = run_chatopenai(
            model_name=self.model_name, system_prompt=self.system_prompt, user_prompt=self.user_prompt
        )

        # Assertions
        self.assertEqual(result, "2 + 2 equals 4.")
        mock_completion.assert_called_once()

        # Check that the correct messages were passed
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]["model"], self.model_name)
        self.assertEqual(len(call_args[1]["messages"]), 2)
        self.assertEqual(call_args[1]["messages"][0]["role"], "system")
        self.assertEqual(call_args[1]["messages"][0]["content"], self.system_prompt)
        self.assertEqual(call_args[1]["messages"][1]["role"], "user")
        self.assertEqual(call_args[1]["messages"][1]["content"], self.user_prompt)

    @patch("run_utils.litellm.completion")
    def test_run_chatopenai_no_system_prompt(self, mock_completion):
        """Test run_chatopenai without system prompt"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_completion.return_value = mock_response

        result = run_chatopenai(model_name=self.model_name, system_prompt=None, user_prompt=self.user_prompt)

        self.assertEqual(result, "Hello!")

        # Check that only user message was passed
        call_args = mock_completion.call_args
        self.assertEqual(len(call_args[1]["messages"]), 1)
        self.assertEqual(call_args[1]["messages"][0]["role"], "user")

    @patch("run_utils.litellm.completion")
    def test_run_chatopenai_json_mode(self, mock_completion):
        """Test run_chatopenai with JSON mode enabled"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "4"}'
        mock_completion.return_value = mock_response

        result = run_chatopenai(
            model_name=self.model_name, system_prompt=self.system_prompt, user_prompt=self.user_prompt, json_mode=True
        )

        self.assertEqual(result, '{"answer": "4"}')

        # Check that response_format was set
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]["response_format"], {"type": "json_object"})

    @patch("run_utils.litellm.completion")
    def test_run_chatopenai_custom_temperature(self, mock_completion):
        """Test run_chatopenai with custom temperature"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_completion.return_value = mock_response

        result = run_chatopenai(
            model_name=self.model_name, system_prompt=self.system_prompt, user_prompt=self.user_prompt, temperature=0.7
        )

        # Check that custom temperature was used
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]["temperature"], 0.7)
        return result

    def test_extract_json_from_response_valid(self):
        """Test extract_json_from_response with valid JSON"""
        response = 'Here is the answer: {"result": "success", "value": 42}'
        result = extract_json_from_response(response)
        self.assertEqual(result, {"result": "success", "value": 42})

    def test_extract_json_from_response_invalid(self):
        """Test extract_json_from_response with invalid JSON"""
        response = "Here is the answer: {invalid json}"
        result = extract_json_from_response(response)
        self.assertIsNone(result)

    def test_extract_json_from_response_no_json(self):
        """Test extract_json_from_response with no JSON"""
        response = "Here is the answer without any JSON"
        result = extract_json_from_response(response)
        self.assertIsNone(result)


class TestRunChatOpenAIIntegration(unittest.TestCase):
    """Integration tests that require actual API calls (disabled by default)"""

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "Skipping integration test - OPENAI_API_KEY not set")
    def test_run_chatopenai_integration(self):
        """Integration test with actual API call"""
        # This test will only run if OPENAI_API_KEY is set
        result = run_chatopenai(
            model_name="gpt-3.5-turbo",
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, World!'",
            temperature=0,
        )

        self.assertIsInstance(result, str)
        self.assertIn("Hello", result)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
