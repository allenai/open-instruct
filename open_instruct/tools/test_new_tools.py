"""Tests for new tools (PythonCodeTool and SerperSearchTool from new_tools.py)."""

import asyncio
import dataclasses
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from open_instruct.tools.new_tools import (
    PythonCodeTool,
    PythonCodeToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
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


class TestSerperSearchToolInit(unittest.TestCase):
    """Tests for SerperSearchTool initialization and properties."""

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_initialization_with_defaults(self):
        """Test tool initializes with correct default values."""
        tool = SerperSearchTool()

        self.assertEqual(tool.num_results, 5)
        self.assertEqual(tool.call_name, "serper_search")
        self.assertEqual(tool.config_name, "serper_search")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_initialization_with_custom_values(self):
        """Test tool initializes with custom values."""
        tool = SerperSearchTool(num_results=10)

        self.assertEqual(tool.num_results, 10)
        self.assertEqual(tool.call_name, "serper_search")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_call_name(self):
        """Test tool call_name is 'serper_search'."""
        tool = SerperSearchTool()
        self.assertEqual(tool.call_name, "serper_search")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = SerperSearchTool()
        self.assertEqual(tool.description, "Google search via the Serper API")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = SerperSearchTool()
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertIn("query", params["required"])
        self.assertEqual(params["properties"]["query"]["description"], "The search query for Google")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = SerperSearchTool()
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "serper_search")
        self.assertEqual(definition["function"]["description"], "Google search via the Serper API")
        self.assertIn("parameters", definition["function"])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing SERPER_API_KEY raises a ValueError on initialization."""
        with self.assertRaises(ValueError) as context:
            SerperSearchTool()

        self.assertEqual(str(context.exception), "Missing SERPER_API_KEY environment variable.")


class TestSerperSearchToolExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for SerperSearchTool execution (async __call__)."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.dict("os.environ", {"SERPER_API_KEY": "test_key"}):
            self.tool = SerperSearchTool(num_results=5)

    async def test_empty_query_returns_error(self):
        """Test that empty query returns an error without calling API."""
        result = await self.tool("")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty query. Please provide a search query.")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)

    async def test_whitespace_only_query_returns_error(self):
        """Test that whitespace-only query returns an error."""
        result = await self.tool("   \n\t  ")

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Empty query. Please provide a search query.")
        self.assertTrue(result.called)

    async def test_none_query_returns_error(self):
        """Test that None query returns an error."""
        result = await self.tool(None)

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.error, "Empty query. Please provide a search query.")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_search_with_organic_results(self, mock_session_class):
        """Test successful search with organic results."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"},
                {"title": "Result 2", "snippet": "This is result 2", "link": "https://example.com/2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("test query")

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("Result 1", result.output)
        self.assertIn("This is result 1", result.output)
        self.assertIn("https://example.com/1", result.output)
        self.assertIn("Result 2", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        self.assertEqual(call_args[0][0], "https://google.serper.dev/search")
        self.assertEqual(call_args[1]["json"]["q"], "test query")
        self.assertEqual(call_args[1]["json"]["num"], 5)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_search_with_answer_box(self, mock_session_class):
        """Test successful search with answer box."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "answerBox": {"answer": "The answer is 42"},
            "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("what is the answer")

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("Direct Answer", result.output)
        self.assertIn("The answer is 42", result.output)
        self.assertIn("Result 1", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_successful_search_with_featured_snippet(self, mock_session_class):
        """Test successful search with featured snippet in answer box."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "answerBox": {"snippet": "This is a featured snippet"},
            "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("what is python")

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("Featured Snippet", result.output)
        self.assertIn("This is a featured snippet", result.output)
        self.assertTrue(result.called)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_no_results_returns_error(self, mock_session_class):
        """Test query with no results returns an error."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("nonexistent query")

        self.assertTrue(result.called)
        self.assertEqual(result.output, "")
        self.assertEqual(result.error, "Query returned no results.")
        self.assertFalse(result.timeout)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_timeout_handling(self, mock_session_class):
        """Test timeout is handled correctly."""
        mock_session = MagicMock()
        mock_session.post.side_effect = asyncio.TimeoutError("Connection timed out")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("test query")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertEqual(result.error, "Timeout after 10 seconds")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_api_connection_error(self, mock_session_class):
        """Test handling of API connection errors."""
        mock_session = MagicMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection refused")
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("test query")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("Connection error", result.error)
        self.assertIn("Connection refused", result.error)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_http_error_handling(self, mock_session_class):
        """Test handling of HTTP errors (e.g., 403, 500)."""
        mock_session = MagicMock()
        mock_session.post.side_effect = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=403, message="Forbidden"
        )
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("test query")

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("HTTP error", result.error)
        self.assertIn("403", result.error)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_custom_num_results(self, mock_session_class):
        """Test that custom num_results is passed to API."""
        tool = SerperSearchTool(num_results=10)

        mock_response = AsyncMock()
        mock_response.json.return_value = {"organic": []}
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        await tool("test query")

        call_args = mock_session.post.call_args
        self.assertEqual(call_args[1]["json"]["num"], 10)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    @patch("open_instruct.tools.new_tools.aiohttp.ClientSession")
    async def test_results_without_snippets_filtered_out(self, mock_session_class):
        """Test that results without snippets are filtered out."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "Result 1", "link": "https://example.com/1"},  # No snippet
                {"title": "Result 2", "snippet": "This has a snippet", "link": "https://example.com/2"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None
        mock_session_class.return_value = mock_session

        result = await self.tool("test query")

        self.assertNotIn("Result 1", result.output)
        self.assertIn("Result 2", result.output)
        self.assertIn("This has a snippet", result.output)


class TestSerperSearchToolConfig(unittest.TestCase):
    """Tests for SerperSearchToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = SerperSearchToolConfig()

        self.assertEqual(config.num_results, 5)

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = SerperSearchToolConfig(num_results=10)

        self.assertEqual(config.num_results, 10)

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_build_creates_tool(self):
        """Test building config creates tool correctly."""
        config = SerperSearchToolConfig(num_results=8)

        tool = config.build()

        self.assertIsInstance(tool, SerperSearchTool)
        self.assertEqual(tool.num_results, 8)

    def test_tool_class_attribute(self):
        """Test tool_class is set to SerperSearchTool."""
        self.assertEqual(SerperSearchToolConfig.tool_class, SerperSearchTool)


if __name__ == "__main__":
    unittest.main()
