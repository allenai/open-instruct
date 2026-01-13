"""Tests for new tools (PythonCodeTool and JinaBrowseTool from new_tools.py)."""

import asyncio
import dataclasses
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from open_instruct.tools.new_tools import (
    JinaBrowseTool,
    JinaBrowseToolConfig,
    PythonCodeTool,
    PythonCodeToolConfig,
    _truncate,
)
from open_instruct.tools.utils import ToolOutput, get_openai_tool_definitions


class TestPythonCodeToolInit(unittest.TestCase):
    """Tests for PythonCodeTool initialization and properties."""

    def test_initialization_with_defaults(self):
        """Test tool initializes with correct default values."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")

        self.assertEqual(tool.api_endpoint, "http://localhost:1212/execute")
        self.assertEqual(tool.timeout, 3)
        self.assertEqual(tool.call_name, "python")
        self.assertEqual(tool.config_name, "python")

    def test_initialization_with_custom_values(self):
        """Test tool initializes with custom values."""
        tool = PythonCodeTool(api_endpoint="http://example.com/run", timeout=10)

        self.assertEqual(tool.api_endpoint, "http://example.com/run")
        self.assertEqual(tool.timeout, 10)
        self.assertEqual(tool.call_name, "python")

    def test_tool_call_name(self):
        """Test tool call_name is 'python'."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.call_name, "python")

    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.description, "Executes Python code and returns printed output.")

    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("code", params["properties"])
        self.assertIn("code", params["required"])
        self.assertEqual(params["properties"]["code"]["description"], "Python code to execute")

    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute")
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "python")
        self.assertEqual(definition["function"]["description"], "Executes Python code and returns printed output.")
        self.assertIn("parameters", definition["function"])


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
        result = await self.tool(None)

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.error, "Empty code. Please provide some code to execute.")

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_execution(self, mock_session_class):
        """Test successful code execution."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"output": "Hello, World!\n", "error": None}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("print('Hello, World!')")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Hello, World!\n")
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

        mock_session.post.assert_called_once_with(
            "http://localhost:1212/execute", json={"code": "print('Hello, World!')", "timeout": 3}
        )

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_execution_with_error_response(self, mock_session_class):
        """Test code execution that returns an error from the API."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"output": "", "error": "NameError: name 'undefined_var' is not defined"}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("print(undefined_var)")

        self.assertTrue(result.called)
        self.assertIn("NameError", result.output)
        self.assertEqual(result.error, "NameError: name 'undefined_var' is not defined")
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_timeout_handling(self, mock_session_class):
        """Test timeout is handled correctly."""
        mock_session = MagicMock()
        mock_session.post.side_effect = asyncio.TimeoutError("Connection timed out")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("import time; time.sleep(100)")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertIn("Timeout after 3 seconds", result.output)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_api_connection_error(self, mock_session_class):
        """Test handling of API connection errors."""
        mock_session = MagicMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection refused")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("print('test')")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("Connection error", result.output)
        self.assertIn("Connection refused", result.output)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_execution_with_output_and_error(self, mock_session_class):
        """Test execution that produces both output and an error."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"output": "Partial output\n", "error": "Some warning or error"}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("print('Partial output'); raise Exception()")

        self.assertTrue(result.called)
        self.assertIn("Partial output", result.output)
        self.assertIn("Some warning or error", result.output)
        self.assertEqual(result.error, "Some warning or error")

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_custom_timeout(self, mock_session_class):
        """Test that custom timeout is passed to API."""
        tool = PythonCodeTool(api_endpoint="http://localhost:1212/execute", timeout=10)

        mock_response = AsyncMock()
        mock_response.json.return_value = {"output": "OK", "error": None}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        await tool("print('test')")

        mock_session.post.assert_called_once_with(
            "http://localhost:1212/execute", json={"code": "print('test')", "timeout": 10}
        )


class TestPythonCodeToolConfig(unittest.TestCase):
    """Tests for PythonCodeToolConfig."""

    def test_config_requires_api_endpoint(self):
        """Test config requires api_endpoint."""
        fields = {f.name: f for f in dataclasses.fields(PythonCodeToolConfig)}
        self.assertIn("api_endpoint", fields)
        self.assertEqual(fields["api_endpoint"].default, dataclasses.MISSING)
        self.assertEqual(fields["api_endpoint"].default_factory, dataclasses.MISSING)

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = PythonCodeToolConfig(api_endpoint="http://example.com/execute", timeout=5)

        self.assertEqual(config.api_endpoint, "http://example.com/execute")
        self.assertEqual(config.timeout, 5)

    def test_build_with_endpoint(self):
        """Test building with api_endpoint creates tool correctly."""
        config = PythonCodeToolConfig(api_endpoint="http://localhost:1212/execute", timeout=5)

        tool = config.build()

        self.assertIsInstance(tool, PythonCodeTool)
        self.assertEqual(tool.api_endpoint, "http://localhost:1212/execute")
        self.assertEqual(tool.timeout, 5)

    def test_tool_class_attribute(self):
        """Test tool_class is set to PythonCodeTool."""
        self.assertEqual(PythonCodeToolConfig.tool_class, PythonCodeTool)


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


class TestJinaBrowseToolInit(unittest.TestCase):
    """Tests for JinaBrowseTool initialization and properties."""

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_initialization_with_defaults(self):
        """Test tool initializes with correct default values."""
        tool = JinaBrowseTool()

        self.assertEqual(tool.timeout, 30)
        self.assertEqual(tool.api_key, "test_key")
        self.assertEqual(tool.call_name, "jina_browse")
        self.assertEqual(tool.config_name, "jina_browse")

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_initialization_with_custom_values(self):
        """Test tool initializes with custom values."""
        tool = JinaBrowseTool(timeout=60)

        self.assertEqual(tool.timeout, 60)
        self.assertEqual(tool.api_key, "test_key")
        self.assertEqual(tool.call_name, "jina_browse")

    @patch.dict("os.environ", {}, clear=True)
    def test_initialization_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with self.assertRaises(ValueError) as context:
            JinaBrowseTool()

        self.assertIn("JINA_API_KEY", str(context.exception))

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_call_name(self):
        """Test tool call_name is 'jina_browse'."""
        tool = JinaBrowseTool()
        self.assertEqual(tool.call_name, "jina_browse")

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = JinaBrowseTool()
        self.assertEqual(
            tool.description, "Fetches and converts webpage content to clean markdown using Jina Reader API"
        )

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = JinaBrowseTool()
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("url", params["properties"])
        self.assertIn("url", params["required"])
        self.assertEqual(params["properties"]["url"]["description"], "The URL of the webpage to fetch")

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = JinaBrowseTool()
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "jina_browse")
        self.assertIn("parameters", definition["function"])


class TestJinaBrowseToolExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for JinaBrowseTool execution (async __call__)."""

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def setUp(self):
        """Set up test fixtures."""
        self.tool = JinaBrowseTool(timeout=30)

    async def test_empty_url_returns_error(self):
        """Test that empty URL returns an error without calling API."""
        result = await self.tool("")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty URL. Please provide a URL to fetch.")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)

    async def test_whitespace_only_url_returns_error(self):
        """Test that whitespace-only URL returns an error."""
        result = await self.tool("   \n\t  ")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty URL. Please provide a URL to fetch.")
        self.assertTrue(result.called)

    async def test_none_url_returns_error(self):
        """Test that None URL returns an error."""
        result = await self.tool(None)

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.error, "Empty URL. Please provide a URL to fetch.")

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_fetch_with_title(self, mock_session_class):
        """Test successful webpage fetch with title."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": {"title": "Example Page", "content": "This is the page content in markdown format."},
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://example.com")

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("Example Page", result.output)
        self.assertIn("This is the page content", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

        # Verify API call
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        self.assertEqual(call_args[0][0], "https://r.jina.ai/https://example.com")
        self.assertIn("Authorization", call_args[1]["headers"])
        self.assertEqual(call_args[1]["headers"]["Authorization"], "Bearer test_key")

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_fetch_without_title(self, mock_session_class):
        """Test successful webpage fetch without title."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"code": 200, "data": {"content": "Just content without a title."}}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://example.com/page")

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("Just content without a title", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_api_error_response(self, mock_session_class):
        """Test handling of Jina API error response."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"code": 400, "message": "Invalid URL format"}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://invalid-url")

        self.assertTrue(result.called)
        self.assertEqual(result.output, "")
        self.assertIn("Jina API error", result.error)
        self.assertIn("Invalid URL format", result.error)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_timeout_handling(self, mock_session_class):
        """Test timeout is handled correctly."""
        mock_session = MagicMock()
        mock_session.get.side_effect = asyncio.TimeoutError("Connection timed out")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://example.com")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertEqual(result.error, "Timeout after 30 seconds")

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_http_error_handling(self, mock_session_class):
        """Test handling of HTTP errors (e.g., 403, 500)."""
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=403, message="Forbidden"
        )
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://example.com")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("HTTP error", result.error)
        self.assertIn("403", result.error)

    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_connection_error_handling(self, mock_session_class):
        """Test handling of connection errors."""
        mock_session = MagicMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection refused")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("https://example.com")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("Connection error", result.error)
        self.assertIn("Connection refused", result.error)

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_custom_timeout(self, mock_session_class):
        """Test that custom timeout is used."""
        tool = JinaBrowseTool(timeout=60)

        mock_response = AsyncMock()
        mock_response.json.return_value = {"code": 200, "data": {"content": "OK"}}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        await tool("https://example.com")

        # Verify ClientSession was created with correct timeout
        call_args = mock_session_class.call_args
        self.assertIsNotNone(call_args[1].get("timeout"))


class TestJinaBrowseToolConfig(unittest.TestCase):
    """Tests for JinaBrowseToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = JinaBrowseToolConfig()

        self.assertEqual(config.timeout, 30)

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = JinaBrowseToolConfig(timeout=60)

        self.assertEqual(config.timeout, 60)

    @patch.dict("os.environ", {"JINA_API_KEY": "env_key"})
    def test_build_creates_tool(self):
        """Test building config creates tool correctly."""
        config = JinaBrowseToolConfig(timeout=45)

        tool = config.build()

        self.assertIsInstance(tool, JinaBrowseTool)
        self.assertEqual(tool.timeout, 45)

    def test_tool_class_attribute(self):
        """Test tool_class is set to JinaBrowseTool."""
        self.assertEqual(JinaBrowseToolConfig.tool_class, JinaBrowseTool)


if __name__ == "__main__":
    unittest.main()
