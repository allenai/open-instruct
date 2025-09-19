import subprocess
import time
import unittest
from unittest.mock import MagicMock, patch

import requests

from open_instruct.tool_utils.tools import MaxCallsExceededTool, PythonCodeTool, Tool, ToolOutput


class TestToolOutput(unittest.TestCase):
    def test_tool_output_creation(self):
        output = ToolOutput(output="test output", called=True, error="test error", timeout=False, runtime=1.5)
        self.assertEqual(output.output, "test output")
        self.assertTrue(output.called)
        self.assertEqual(output.error, "test error")
        self.assertFalse(output.timeout)
        self.assertEqual(output.runtime, 1.5)
        self.assertEqual(output.start_str, "<output>\n")
        self.assertEqual(output.end_str, "\n</output>")

    def test_tool_output_custom_delimiters(self):
        output = ToolOutput(
            output="test",
            called=False,
            error="",
            timeout=True,
            runtime=0.0,
            start_str="<custom_start>",
            end_str="<custom_end>",
        )
        self.assertEqual(output.start_str, "<custom_start>")
        self.assertEqual(output.end_str, "<custom_end>")


class TestTool(unittest.TestCase):
    def test_tool_initialization(self):
        tool = Tool(start_str="<start>", end_str="<end>")
        self.assertEqual(tool.start_str, "<start>")
        self.assertEqual(tool.end_str, "<end>")

    def test_tool_call_not_implemented(self):
        tool = Tool(start_str="<start>", end_str="<end>")
        with self.assertRaises(NotImplementedError) as context:
            tool("test prompt")
        self.assertIn("Subclasses must implement this method", str(context.exception))


class TestMaxCallsExceededTool(unittest.TestCase):
    def test_max_calls_exceeded_output(self):
        tool = MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")
        result = tool("any prompt")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Max tool calls exceeded.")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)


class TestPythonCodeTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the tool server for integration tests."""
        cls.server_process = None
        cls.use_real_server = False  # Set to True to test with real server

        if cls.use_real_server:
            # Start the server in a subprocess
            cls.server_process = subprocess.Popen(
                ["uv", "run", "uvicorn", "tool_server:app", "--host", "0.0.0.0", "--port", "1212"],
                cwd="open_instruct/tool_utils",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Create new process group
            )
            # Wait for server to start
            time.sleep(3)
            cls.api_endpoint = "http://localhost:1212/execute"
        else:
            cls.api_endpoint = "http://test-api.com/execute"

    @classmethod
    def tearDownClass(cls):
        """Stop the tool server."""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()

    def setUp(self):
        self.api_endpoint = self.__class__.api_endpoint
        self.tool = PythonCodeTool(api_endpoint=self.api_endpoint, start_str="<code>", end_str="</code>")

    def test_initialization(self):
        self.assertEqual(self.tool.api_endpoint, self.api_endpoint)
        self.assertEqual(self.tool.start_str, "<code>")
        self.assertEqual(self.tool.end_str, "</code>")

    def test_no_code_blocks(self):
        prompt = "This is a prompt without any code blocks."
        result = self.tool(prompt)

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)

    @patch("requests.post")
    def test_successful_code_execution(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "Hello, World!", "error": None}
        mock_post.return_value = mock_response

        prompt = """Let me calculate this.
<code>
print("Hello, World!")
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertEqual(result.output, "Hello, World!")
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

        # Verify API was called correctly
        mock_post.assert_called_once_with(
            self.api_endpoint, json={"code": 'print("Hello, World!")', "timeout": 3}, timeout=3
        )

    @patch("requests.post")
    def test_code_execution_with_error(self, mock_post):
        # Mock API response with error
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "", "error": "SyntaxError: invalid syntax"}
        mock_post.return_value = mock_response

        prompt = """<code>
print("unclosed string
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("SyntaxError: invalid syntax", result.output)
        self.assertEqual(result.error, "SyntaxError: invalid syntax")
        self.assertFalse(result.timeout)

    @patch("requests.post")
    def test_timeout_handling(self, mock_post):
        # Mock timeout exception
        mock_post.side_effect = requests.Timeout("Request timed out")

        prompt = """<code>
import time
time.sleep(10)
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("Timeout after 3 seconds", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.timeout)

    @patch("requests.post")
    def test_api_error_handling(self, mock_post):
        # Mock general API error
        mock_post.side_effect = Exception("API connection failed")

        prompt = """<code>
print("test")
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("Error calling API: API connection failed", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)

    @patch("requests.post")
    def test_multiple_code_blocks_uses_last(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "Second block output", "error": None}
        mock_post.return_value = mock_response

        prompt = """First code block:
<code>
print("First block")
</code>

Second code block:
<code>
print("Second block")
</code>"""

        result = self.tool(prompt)

        # Should only execute the last code block
        mock_post.assert_called_once_with(
            self.api_endpoint, json={"code": 'print("Second block")', "timeout": 3}, timeout=3
        )
        self.assertEqual(result.output, "Second block output")

    def test_code_block_with_backticks_ignored(self):
        # Test that code blocks preceded by backticks are ignored
        prompt = """Here's some inline code: `<code>print("ignored")</code>`

And here's actual code:
<code>
print("executed")
</code>"""

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"output": "executed", "error": None}
            mock_post.return_value = mock_response

            self.tool(prompt)

            # Should only find and execute the non-backticked code block
            mock_post.assert_called_once_with(
                self.api_endpoint, json={"code": 'print("executed")', "timeout": 3}, timeout=3
            )


class TestPythonCodeToolIntegration(unittest.TestCase):
    """Integration tests that use the real tool server."""

    @classmethod
    def setUpClass(cls):
        """Start the real tool server for integration tests."""
        # Start the server in a subprocess
        cls.server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "tool_server:app", "--host", "0.0.0.0", "--port", "1213"],
            cwd="open_instruct/tool_utils",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Create new process group
        )
        # Wait for server to start
        time.sleep(3)
        cls.api_endpoint = "http://localhost:1213/execute"

    @classmethod
    def tearDownClass(cls):
        """Stop the tool server."""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()

    def setUp(self):
        self.tool = PythonCodeTool(api_endpoint=self.api_endpoint, start_str="<code>", end_str="</code>")

    def test_real_code_execution(self):
        """Test actual code execution with the real server."""
        prompt = """<code>
print("Hello from integration test!")
print(2 + 2)
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("Hello from integration test!", result.output)
        self.assertIn("4", result.output)
        self.assertFalse(result.timeout)
        self.assertEqual(result.error, "")

    def test_real_code_with_error(self):
        """Test code with syntax error using real server."""
        prompt = """<code>
print("unclosed string
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertTrue(len(result.error) > 0 or "Error" in result.output)

    def test_real_timeout(self):
        """Test timeout handling with real server."""
        prompt = """<code>
import time
time.sleep(5)
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertIn("Timeout", result.output)


if __name__ == "__main__":
    # Run only unit tests by default
    # To run integration tests, use: python test_tools.py TestPythonCodeToolIntegration
    unittest.main()
