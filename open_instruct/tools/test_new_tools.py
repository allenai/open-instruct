"""Tests for new tools (PythonCodeTool from new_tools.py)."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from open_instruct.tools.new_tools import PythonCodeTool, PythonCodeToolConfig, _truncate
from open_instruct.tools.utils import Tool, ToolOutput


class TestPythonCodeToolInit(unittest.TestCase):
    """Tests for PythonCodeTool initialization and properties."""

    def test_initialization_with_defaults(self):
        """Test tool initializes with correct default values."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")

        self.assertEqual(tool.api_endpoint, "http://localhost:1212/execute")
        self.assertEqual(tool.timeout, 3)
        self.assertIsNone(tool.override_name)

    def test_initialization_with_custom_values(self):
        """Test tool initializes with custom values."""
        tool = PythonCodeTool(api_endpoint="http://example.com/run", timeout=10, override_name="custom_python")

        self.assertEqual(tool.api_endpoint, "http://example.com/run")
        self.assertEqual(tool.timeout, 10)
        self.assertEqual(tool.override_name, "custom_python")

    def test_default_tool_function_name(self):
        """Test default tool function name is 'python'."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.tool_function_name, "python")

    def test_override_tool_function_name(self):
        """Test overriding tool function name."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute", override_name="code")
        self.assertEqual(tool.tool_function_name, "code")

    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.tool_description, "Executes Python code and returns printed output.")

    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correctly inferred."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        params = tool.tool_parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("code", params["properties"])
        self.assertIn("code", params["required"])
        # Check description from Annotated Field
        self.assertEqual(params["properties"]["code"]["description"], "Python code to execute")

    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        definitions = tool.get_openai_tool_definitions()

        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["type"], "function")
        self.assertEqual(definitions[0]["function"]["name"], "python")
        self.assertEqual(definitions[0]["function"]["description"], "Executes Python code and returns printed output.")
        self.assertIn("parameters", definitions[0]["function"])

    def test_get_tool_names(self):
        """Test get_tool_names returns correct name."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        names = tool.get_tool_names()

        self.assertEqual(names, ["python"])

    def test_get_tool_names_with_override(self):
        """Test get_tool_names returns overridden name."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute", override_name="execute")
        names = tool.get_tool_names()

        self.assertEqual(names, ["execute"])


class TestPythonCodeToolExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for PythonCodeTool execution (async __call__)."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute", timeout=3)

    async def test_empty_code_returns_error(self):
        """Test that empty code returns an error without calling API."""
        result = await self.tool("")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty code. Please provide some code to execute.")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)

    async def test_whitespace_only_code_returns_error(self):
        """Test that whitespace-only code returns an error."""
        result = await self.tool("   \n\t  ")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty code. Please provide some code to execute.")
        self.assertTrue(result.called)

    async def test_none_code_returns_error(self):
        """Test that None code returns an error."""
        # This shouldn't happen in practice due to type hints, but test defensively
        result = await self.tool(None)

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.error, "Empty code. Please provide some code to execute.")

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_successful_execution(self, mock_client_class):
        """Test successful code execution."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "Hello, World!\n", "error": None}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.tool("print('Hello, World!')")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Hello, World!\n")
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

        # Verify API was called with correct parameters
        mock_client.post.assert_called_once_with(
            "http://localhost:1212/execute", json={"code": "print('Hello, World!')", "timeout": 3}, timeout=3
        )

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_execution_with_error_response(self, mock_client_class):
        """Test code execution that returns an error from the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "", "error": "NameError: name 'undefined_var' is not defined"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.tool("print(undefined_var)")

        self.assertTrue(result.called)
        self.assertIn("NameError", result.output)
        self.assertEqual(result.error, "NameError: name 'undefined_var' is not defined")
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_timeout_handling(self, mock_client_class):
        """Test timeout is handled correctly."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.tool("import time; time.sleep(100)")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertIn("Timeout after 3 seconds", result.output)

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_api_connection_error(self, mock_client_class):
        """Test handling of API connection errors."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.tool("print('test')")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("Error calling API", result.output)
        self.assertIn("Connection refused", result.output)

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_execution_with_output_and_error(self, mock_client_class):
        """Test execution that produces both output and an error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "Partial output\n", "error": "Some warning or error"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.tool("print('Partial output'); raise Exception()")

        self.assertTrue(result.called)
        self.assertIn("Partial output", result.output)
        self.assertIn("Some warning or error", result.output)
        self.assertEqual(result.error, "Some warning or error")

    @patch("open_instruct.tools.new_tools.httpx.AsyncClient")
    async def test_custom_timeout(self, mock_client_class):
        """Test that custom timeout is passed to API."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute", timeout=10)

        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "OK", "error": None}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        await tool("print('test')")

        mock_client.post.assert_called_once_with(
            "http://localhost:1212/execute", json={"code": "print('test')", "timeout": 10}, timeout=10
        )


class TestPythonCodeToolConfig(unittest.TestCase):
    """Tests for PythonCodeToolConfig."""

    def test_config_defaults(self):
        """Test config has correct default values."""
        config = PythonCodeToolConfig()

        self.assertIsNone(config.api_endpoint)
        self.assertEqual(config.timeout, 3)
        self.assertIsNone(config.override_name)

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = PythonCodeToolConfig(api_endpoint="http://example.com/execute", timeout=5, override_name="code_exec")

        self.assertEqual(config.api_endpoint, "http://example.com/execute")
        self.assertEqual(config.timeout, 5)
        self.assertEqual(config.override_name, "code_exec")

    def test_build_without_endpoint_raises(self):
        """Test building without api_endpoint raises ValueError."""
        config = PythonCodeToolConfig()

        with self.assertRaises(ValueError) as context:
            config.build()

        self.assertIn("api_endpoint must be set", str(context.exception))

    def test_build_with_endpoint(self):
        """Test building with api_endpoint creates tool correctly."""
        config = PythonCodeToolConfig(api_endpoint="http://localhost:1212/execute", timeout=5)

        tool = config.build()

        self.assertIsInstance(tool, PythonCodeTool)
        self.assertEqual(tool.api_endpoint, "http://localhost:1212/execute")
        self.assertEqual(tool.timeout, 5)

    def test_build_with_override_name(self):
        """Test building with override_name sets it correctly."""
        config = PythonCodeToolConfig(api_endpoint="http://localhost:1212/execute", override_name="execute_code")

        tool = config.build()

        self.assertEqual(tool.tool_function_name, "execute_code")

    def test_tool_class_attribute(self):
        """Test tool_class is set to PythonCodeTool."""
        self.assertEqual(PythonCodeToolConfig.tool_class, PythonCodeTool)


class TestToolSubclassRequirements(unittest.TestCase):
    """Tests for Tool subclass requirements."""

    def test_missing_default_tool_name_raises(self):
        """Test that subclass without default_tool_name raises TypeError."""
        with self.assertRaises(TypeError) as context:

            class BadTool(Tool):
                default_description = "A tool"

                async def __call__(self):
                    pass

        self.assertIn("default_tool_name", str(context.exception))

    def test_missing_default_description_raises(self):
        """Test that subclass without default_description raises TypeError."""
        with self.assertRaises(TypeError) as context:

            class BadTool(Tool):
                default_tool_name = "bad"

                async def __call__(self):
                    pass

        self.assertIn("default_description", str(context.exception))

    def test_valid_subclass_works(self):
        """Test that properly defined subclass works."""

        class GoodTool(Tool):
            default_tool_name = "good"
            default_description = "A good tool"

            async def __call__(self):
                pass

        tool = GoodTool()
        self.assertEqual(tool.tool_function_name, "good")
        self.assertEqual(tool.tool_description, "A good tool")


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions in new_tools.py."""

    def test_truncate_short_text(self):
        """Test _truncate doesn't modify short text."""
        short_text = "Hello, World!"
        result = _truncate(short_text, max_length=500)
        self.assertEqual(result, short_text)

    def test_truncate_long_text(self):
        """Test _truncate truncates long text with ellipsis."""
        long_text = "a" * 600
        result = _truncate(long_text, max_length=500)

        self.assertEqual(len(result), 500 + len("... [100 more chars]"))
        self.assertTrue(result.startswith("a" * 500))
        self.assertTrue(result.endswith("... [100 more chars]"))

    def test_truncate_exact_length(self):
        """Test _truncate at exact max_length boundary."""
        text = "a" * 500
        result = _truncate(text, max_length=500)
        self.assertEqual(result, text)

    def test_truncate_custom_max_length(self):
        """Test _truncate with custom max_length."""
        text = "Hello, World! This is a test."
        result = _truncate(text, max_length=10)

        self.assertTrue(result.startswith("Hello, Wor"))
        self.assertIn("more chars", result)


if __name__ == "__main__":
    unittest.main()
