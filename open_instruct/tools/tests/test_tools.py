import subprocess
import time
import unittest

from open_instruct.tools.base import ToolOutput
from open_instruct.tools.tools import MaxCallsExceededTool, PythonCodeTool, PythonCodeToolConfig


class TestToolOutput(unittest.TestCase):
    def test_tool_output_creation(self):
        output = ToolOutput(output="test output", called=True, error="test error", timeout=False, runtime=1.5)
        self.assertEqual(output.output, "test output")
        self.assertTrue(output.called)
        self.assertEqual(output.error, "test error")
        self.assertFalse(output.timeout)
        self.assertEqual(output.runtime, 1.5)


class TestMaxCallsExceededTool(unittest.TestCase):
    def test_max_calls_exceeded_output(self):
        tool = MaxCallsExceededTool()
        result = tool()

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Max tool calls exceeded.")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)


class TestPythonCodeTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the tool server for tests."""
        # Start the server in a subprocess
        cls.server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "1212"],
            cwd="open_instruct/tools/code_server",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Create new process group
        )
        # Wait for server to start
        time.sleep(3)
        cls.api_endpoint = "http://localhost:1212/execute"

    @classmethod
    def tearDownClass(cls):
        """Stop the tool server."""
        if not cls.server_process:
            return
        cls.server_process.terminate()
        try:
            cls.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server_process.kill()
            cls.server_process.wait()

    def setUp(self):
        self.api_endpoint = self.__class__.api_endpoint
        self.tool = PythonCodeTool(api_endpoint=self.api_endpoint)

    def test_initialization(self):
        self.assertEqual(self.tool.api_endpoint, self.api_endpoint)
        self.assertEqual(self.tool.tool_function_name, "python")

    def test_from_config(self):
        config = PythonCodeToolConfig(api_endpoint=self.api_endpoint, timeout_seconds=5)
        tool = PythonCodeTool.from_config(config)
        self.assertEqual(tool.api_endpoint, self.api_endpoint)
        self.assertEqual(tool.timeout_seconds, 5)

    def test_successful_code_execution(self):
        result = self.tool(text='print("Hello, World!")')

        self.assertTrue(result.called)
        self.assertIn("Hello, World!", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

    def test_code_execution_with_error(self):
        result = self.tool(text='print("unclosed string')

        self.assertTrue(result.called)
        self.assertTrue("SyntaxError" in result.output or len(result.error) > 0)
        self.assertFalse(result.timeout)

    def test_timeout_handling(self):
        result = self.tool(text="import time\ntime.sleep(10)")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout or "Timeout" in result.output or "timeout" in result.error)
        self.assertLess(result.runtime, 10)  # Should timeout before 10 seconds

    def test_computation(self):
        result = self.tool(text='result = 5 * 7 + 3\nprint(f"The result is {result}")')

        self.assertTrue(result.called)
        self.assertIn("The result is 38", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)


if __name__ == "__main__":
    unittest.main()
