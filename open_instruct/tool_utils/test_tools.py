import subprocess
import time
import unittest

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
        """Start the tool server for tests."""
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

    def test_successful_code_execution(self):
        prompt = """Let me calculate this.
<code>
print("Hello, World!")
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("Hello, World!", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

    def test_code_execution_with_error(self):
        prompt = """<code>
print("unclosed string
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertTrue("SyntaxError" in result.output or len(result.error) > 0)
        self.assertFalse(result.timeout)

    def test_timeout_handling(self):
        prompt = """<code>
import time
time.sleep(10)
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertTrue(result.timeout or "Timeout" in result.output or "timeout" in result.error)
        self.assertLess(result.runtime, 10)  # Should timeout before 10 seconds

    def test_computation(self):
        prompt = """<code>
result = 5 * 7 + 3
print(f"The result is {result}")
</code>"""

        result = self.tool(prompt)

        self.assertTrue(result.called)
        self.assertIn("The result is 38", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)

    def test_multiple_code_blocks_uses_last(self):
        prompt = """First code block:
<code>
print("First block")
</code>

Second code block:
<code>
print("Second block")
</code>"""

        result = self.tool(prompt)
        self.assertTrue(result.called)
        self.assertIn("Second block", result.output)
        self.assertNotIn("First block", result.output)

    def test_code_block_with_backticks_ignored(self):
        prompt = """Here's some inline code: `<code>print("ignored")</code>`

And here's actual code:
<code>
print("executed")
</code>"""

        result = self.tool(prompt)
        self.assertTrue(result.called)
        self.assertIn("executed", result.output)
        self.assertNotIn("ignored", result.output)


if __name__ == "__main__":
    unittest.main()
