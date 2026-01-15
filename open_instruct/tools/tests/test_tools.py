"""Tests for tools (PythonCodeTool, JinaBrowseTool, S2SearchTool, and SerperSearchTool from tools.py)."""

import asyncio
import dataclasses
import unittest
from unittest.mock import patch

from parameterized import parameterized

from open_instruct.tools.tools import (
    JinaBrowseTool,
    JinaBrowseToolConfig,
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
    _truncate,
)
from open_instruct.tools.utils import ParsedToolConfig, ToolOutput, ToolsConfig, get_openai_tool_definitions


class TestPythonCodeToolInit(unittest.TestCase):
    """Tests for PythonCodeTool initialization and properties."""

    @parameterized.expand(
        [
            ("defaults", "python", "http://localhost:1212/execute", None, 3),
            ("custom_values", "code", "http://example.com/run", 10, 10),
        ]
    )
    def test_initialization(self, name, call_name, api_endpoint, timeout, expected_timeout):
        """Test tool initializes with correct values."""
        kwargs = {"call_name": call_name, "api_endpoint": api_endpoint}
        if timeout is not None:
            kwargs["timeout"] = timeout
        tool = PythonCodeTool(**kwargs)

        self.assertEqual(tool.api_endpoint, api_endpoint)
        self.assertEqual(tool.timeout, expected_timeout)
        self.assertEqual(tool.call_name, call_name)

    @parameterized.expand([("python",), ("code",)])
    def test_tool_call_name(self, call_name):
        """Test tool call_name can be customized."""
        tool = PythonCodeTool(call_name=call_name, api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.call_name, call_name)

    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        self.assertEqual(tool.description, "Executes Python code and returns printed output.")

    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("code", params["properties"])
        self.assertIn("code", params["required"])
        self.assertEqual(params["properties"]["code"]["description"], "Python code to execute")

    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "python")
        self.assertEqual(definition["function"]["description"], "Executes Python code and returns printed output.")
        self.assertIn("parameters", definition["function"])

    @parameterized.expand([("custom_name", "code", "code"), ("default_name", None, "python")])
    def test_call_name_via_config(self, name, custom_call_name, expected_call_name):
        """Test that call_name can be set when instantiating tools from config."""
        config = PythonCodeToolConfig(api_endpoint="http://example.com/execute")
        # Instantiate tool directly with config values
        call_name = custom_call_name if custom_call_name else config.tool_class.config_name
        tool = PythonCodeTool(call_name=call_name, api_endpoint=config.api_endpoint, timeout=config.timeout)
        self.assertEqual(tool.call_name, expected_call_name)

        # Verify OpenAI definition uses the correct name
        definition = get_openai_tool_definitions(tool)
        self.assertEqual(definition["function"]["name"], expected_call_name)


class TestPythonCodeToolExecution(unittest.TestCase):
    """Tests for PythonCodeTool execution (async execute)."""

    def setUp(self):
        """Set up test fixture."""
        self.tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute", timeout=3)

    @parameterized.expand([("empty", ""), ("whitespace", "   \n\t  "), ("none", None)])
    def test_empty_code_returns_error(self, name, code_input):
        """Test that empty/whitespace/None code returns an error without calling API."""
        result = asyncio.run(self.tool(code_input))

        expected = ToolOutput(
            output="", error="Empty code. Please provide some code to execute.", called=True, timeout=False, runtime=0
        )
        self.assertEqual(result, expected)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_successful_execution(self, mock_api_request):
        """Test successful code execution."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "Hello, World!\n", "error": None})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool("print('Hello, World!')"))

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Hello, World!\n")
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_execution_with_error_response(self, mock_api_request):
        """Test code execution that returns an error from the API."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "", "error": "NameError: name 'undefined_var' is not defined"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool("print(undefined_var)"))

        self.assertTrue(result.called)
        self.assertIn("NameError", result.output)
        self.assertEqual(result.error, "NameError: name 'undefined_var' is not defined")
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_timeout_handling(self, mock_api_request):
        """Test timeout is handled correctly."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(error="Timeout after 3 seconds", timed_out=True)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool("import time; time.sleep(100)"))

        self.assertTrue(result.called)
        self.assertTrue(result.timeout)
        self.assertIn("Timeout after 3 seconds", result.output)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_api_connection_error(self, mock_api_request):
        """Test handling of API connection errors."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(error="Connection error: Connection refused")

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool("print('test')"))

        self.assertTrue(result.called)
        self.assertFalse(result.timeout)
        self.assertIn("Connection error", result.output)
        self.assertIn("Connection refused", result.output)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_execution_with_output_and_error(self, mock_api_request):
        """Test execution that produces both output and an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "Partial output\n", "error": "Some warning or error"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool("print('Partial output'); raise Exception()"))

        self.assertTrue(result.called)
        self.assertIn("Partial output", result.output)
        self.assertIn("Some warning or error", result.output)
        self.assertEqual(result.error, "Some warning or error")

    @patch("open_instruct.tools.tools.make_api_request")
    def test_custom_timeout(self, mock_api_request):
        """Test that custom timeout is passed to API."""
        from open_instruct.tools.utils import APIResponse

        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute", timeout=10)

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "OK", "error": None})

        mock_api_request.side_effect = mock_response

        asyncio.run(tool("print('test')"))

        # Verify the timeout was passed correctly
        mock_api_request.assert_called_once()
        call_args = mock_api_request.call_args
        self.assertEqual(call_args[1]["timeout_seconds"], 10)


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

    def test_config_values_used_for_tool(self):
        """Test config values can be used to create tool correctly."""
        config = PythonCodeToolConfig(api_endpoint="http://localhost:1212/execute", timeout=5)

        # Create tool using config values (as build_remote does internally)
        tool = PythonCodeTool(
            call_name=config.tool_class.config_name, api_endpoint=config.api_endpoint, timeout=config.timeout
        )

        self.assertIsInstance(tool, PythonCodeTool)
        self.assertEqual(tool.api_endpoint, "http://localhost:1212/execute")
        self.assertEqual(tool.timeout, 5)

    def test_tool_class_attribute(self):
        """Test tool_class is set to PythonCodeTool."""
        self.assertEqual(PythonCodeToolConfig.tool_class, PythonCodeTool)


class TestTruncateHelper(unittest.TestCase):
    """Tests for _truncate helper function."""

    @parameterized.expand(
        [
            ("short_text", "Hello, World!", 500, "Hello, World!", False),
            ("long_text", "a" * 600, 500, "a" * 500 + "... [100 more chars]", True),
            ("exact_boundary", "a" * 500, 500, "a" * 500, False),
            ("custom_max_length", "Hello, World! This is a test.", 10, "Hello, Wor", True),
        ]
    )
    def test_truncate(self, name, text, max_length, expected_result, should_truncate):
        """Test _truncate with various inputs."""
        result = _truncate(text, max_length=max_length)

        if should_truncate:
            self.assertTrue(result.startswith(text[:max_length]))
            self.assertIn("more chars", result)
            if text == "a" * 600:
                self.assertEqual(len(result), 500 + len("... [100 more chars]"))
        else:
            self.assertEqual(result, expected_result)


class TestJinaBrowseToolInit(unittest.TestCase):
    """Tests for JinaBrowseTool initialization and properties."""

    @parameterized.expand([("defaults", None, 30), ("custom_values", 60, 60)])
    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_initialization(self, name, timeout, expected_timeout):
        """Test tool initializes with correct values."""
        kwargs = {"call_name": "browse"}
        if timeout is not None:
            kwargs["timeout"] = timeout
        tool = JinaBrowseTool(**kwargs)

        self.assertEqual(tool.timeout, expected_timeout)
        self.assertEqual(tool.call_name, "browse")
        self.assertEqual(tool.config_name, "jina_browse")
        self.assertEqual(tool.api_key, "test_key")

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = JinaBrowseTool(call_name="browse")
        self.assertEqual(
            tool.description, "Fetches and converts webpage content to clean markdown using Jina Reader API"
        )

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = JinaBrowseTool(call_name="browse")
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("url", params["properties"])
        self.assertIn("url", params["required"])
        self.assertEqual(params["properties"]["url"]["description"], "The URL of the webpage to fetch")

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = JinaBrowseTool(call_name="browse")
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "browse")
        self.assertIn("parameters", definition["function"])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing JINA_API_KEY raises a ValueError on initialization."""
        with self.assertRaisesRegex(ValueError, "Missing JINA_API_KEY environment variable."):
            JinaBrowseTool(call_name="browse")


class TestJinaBrowseToolExecution(unittest.TestCase):
    """Tests for JinaBrowseTool execution (async execute)."""

    def setUp(self):
        """Set up test fixture."""
        patcher = patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tool = JinaBrowseTool(call_name="browse", timeout=30)

    @parameterized.expand([("empty", ""), ("whitespace", "   \n\t  "), ("none", None)])
    def test_empty_url_returns_error(self, name, url_input):
        """Test that empty/whitespace/None URL returns an error without calling API."""
        result = asyncio.run(self.tool.execute(url_input))

        expected = ToolOutput(
            output="", error="Empty URL. Please provide a URL to fetch.", called=True, timeout=False, runtime=0
        )
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            (
                "with_title",
                {"code": 200, "data": {"title": "Example Page", "content": "This is the page content."}},
                ["# Example Page", "This is the page content."],
            ),
            (
                "without_title",
                {"code": 200, "data": {"content": "Just content without a title."}},
                ["Just content without a title."],
            ),
        ]
    )
    @patch("open_instruct.tools.tools.make_api_request")
    def test_successful_fetch(self, name, api_data, expected_in_output, mock_api_request):
        """Test successful webpage fetch with various response types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data=api_data)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("https://example.com"))

        self.assertIsInstance(result, ToolOutput)
        for expected in expected_in_output:
            self.assertIn(expected, result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_jina_api_error_response(self, mock_api_request):
        """Test handling of Jina API error response (code != 200)."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"code": 400, "message": "Invalid URL format"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("https://invalid-url"))

        self.assertTrue(result.called)
        self.assertIn("Jina API error", result.error)
        self.assertIn("Invalid URL format", result.error)
        self.assertIn("Jina API error", result.output)  # Error in output for model feedback

    @parameterized.expand(
        [
            ("timeout", {"error": "Timeout after 30 seconds", "timed_out": True}, True, "Timeout after 30 seconds"),
            (
                "connection_error",
                {"error": "Connection error: Connection refused", "timed_out": False},
                False,
                "Connection error",
            ),
            ("http_error", {"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ]
    )
    @patch("open_instruct.tools.tools.make_api_request")
    def test_error_handling(self, name, api_response, expected_timeout, expected_error_contains, mock_api_request):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("https://example.com"))

        self.assertTrue(result.called)
        self.assertEqual(result.timeout, expected_timeout)
        self.assertIn(expected_error_contains, result.error)
        self.assertIn(expected_error_contains, result.output)  # Error message also in output for model feedback

    @patch("open_instruct.tools.tools.make_api_request")
    def test_uses_get_method(self, mock_api_request):
        """Test that JinaBrowseTool uses GET method."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"code": 200, "data": {"content": "OK"}})

        mock_api_request.side_effect = mock_response

        asyncio.run(self.tool.execute("https://example.com"))

        mock_api_request.assert_called_once()
        call_args = mock_api_request.call_args
        self.assertEqual(call_args[1]["method"], "GET")
        self.assertIn("r.jina.ai", call_args[1]["url"])


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

    def test_tool_class_attribute(self):
        """Test tool_class is set to JinaBrowseTool."""
        self.assertEqual(JinaBrowseToolConfig.tool_class, JinaBrowseTool)


class TestS2SearchToolInit(unittest.TestCase):
    """Tests for S2SearchTool initialization and properties."""

    @parameterized.expand([("defaults", None, 10), ("custom_values", 5, 5)])
    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_initialization(self, name, num_results, expected_num_results):
        """Test tool initializes with correct values."""
        kwargs = {"call_name": "s2"}
        if num_results is not None:
            kwargs["num_results"] = num_results
        tool = S2SearchTool(**kwargs)

        self.assertEqual(tool.num_results, expected_num_results)
        self.assertEqual(tool.call_name, "s2")
        self.assertEqual(tool.config_name, "s2_search")

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = S2SearchTool(call_name="s2")
        self.assertIn("Semantic Scholar", tool.description)

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = S2SearchTool(call_name="s2")
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertIn("query", params["required"])

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = S2SearchTool(call_name="s2")
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "s2")
        self.assertIn("parameters", definition["function"])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing S2_API_KEY raises a ValueError on initialization."""
        with self.assertRaisesRegex(ValueError, "Missing S2_API_KEY environment variable."):
            S2SearchTool(call_name="s2")


class TestS2SearchToolExecution(unittest.TestCase):
    """Tests for S2SearchTool execution (async execute)."""

    def setUp(self):
        """Set up test fixture."""
        patcher = patch.dict("os.environ", {"S2_API_KEY": "test_key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tool = S2SearchTool(call_name="s2", num_results=10)

    @parameterized.expand([("empty", ""), ("whitespace", "   \n\t  "), ("none", None)])
    def test_empty_query_returns_error(self, name, query_input):
        """Test that empty/whitespace/None query returns an error without calling API."""
        result = asyncio.run(self.tool.execute(query_input))

        expected = ToolOutput(
            output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
        )
        self.assertEqual(result, expected)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_successful_search(self, mock_api_request):
        """Test successful search with results."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(
                data={
                    "data": [
                        {"snippet": {"text": "First research paper snippet."}},
                        {"snippet": {"text": "Second research paper snippet."}},
                    ]
                }
            )

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("machine learning"))

        self.assertIsInstance(result, ToolOutput)
        self.assertIn("First research paper snippet", result.output)
        self.assertIn("Second research paper snippet", result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"data": []})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("nonexistent query"))

        self.assertTrue(result.called)
        self.assertEqual(result.error, "Query returned no results.")
        self.assertEqual(result.output, result.error)  # Error in output for model feedback
        self.assertFalse(result.timeout)

    @parameterized.expand(
        [
            ("timeout", {"error": "Timeout after 60 seconds", "timed_out": True}, True, "Timeout after 60 seconds"),
            (
                "connection_error",
                {"error": "Connection error: Connection refused", "timed_out": False},
                False,
                "Connection error",
            ),
            ("http_error", {"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ]
    )
    @patch("open_instruct.tools.tools.make_api_request")
    def test_error_handling(self, name, api_response, expected_timeout, expected_error_contains, mock_api_request):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("test query"))

        self.assertTrue(result.called)
        self.assertEqual(result.timeout, expected_timeout)
        self.assertIn(expected_error_contains, result.error)
        self.assertIn(expected_error_contains, result.output)  # Error message also in output for model feedback

    @patch("open_instruct.tools.tools.make_api_request")
    def test_uses_get_method(self, mock_api_request):
        """Test that S2SearchTool uses GET method."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"data": []})

        mock_api_request.side_effect = mock_response

        asyncio.run(self.tool.execute("test query"))

        mock_api_request.assert_called_once()
        call_args = mock_api_request.call_args
        self.assertEqual(call_args[1]["method"], "GET")


class TestS2SearchToolConfig(unittest.TestCase):
    """Tests for S2SearchToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = S2SearchToolConfig()
        self.assertEqual(config.num_results, 10)
        self.assertEqual(config.timeout, 60)

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = S2SearchToolConfig(num_results=5, timeout=30)
        self.assertEqual(config.num_results, 5)
        self.assertEqual(config.timeout, 30)

    def test_tool_class_attribute(self):
        """Test tool_class is set to S2SearchTool."""
        self.assertEqual(S2SearchToolConfig.tool_class, S2SearchTool)


class TestSerperSearchToolInit(unittest.TestCase):
    """Tests for SerperSearchTool initialization and properties."""

    @parameterized.expand([("defaults", None, 5), ("custom_values", 10, 10)])
    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_initialization(self, name, num_results, expected_num_results):
        """Test tool initializes with correct values."""
        kwargs = {"call_name": "search"}
        if num_results is not None:
            kwargs["num_results"] = num_results
        tool = SerperSearchTool(**kwargs)

        self.assertEqual(tool.num_results, expected_num_results)
        self.assertEqual(tool.call_name, "search")
        self.assertEqual(tool.config_name, "serper_search")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = SerperSearchTool(call_name="search")
        self.assertEqual(tool.description, "Google search via the Serper API")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = SerperSearchTool(call_name="search")
        params = tool.parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertIn("query", params["required"])
        self.assertEqual(params["properties"]["query"]["description"], "The search query for Google")

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = SerperSearchTool(call_name="search")
        definition = get_openai_tool_definitions(tool)

        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["function"]["name"], "search")
        self.assertEqual(definition["function"]["description"], "Google search via the Serper API")
        self.assertIn("parameters", definition["function"])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing SERPER_API_KEY raises a ValueError on initialization."""
        with self.assertRaisesRegex(ValueError, "Missing SERPER_API_KEY environment variable."):
            SerperSearchTool(call_name="search")


class TestSerperSearchToolExecution(unittest.TestCase):
    """Tests for SerperSearchTool execution (async execute)."""

    def setUp(self):
        """Set up test fixture."""
        patcher = patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tool = SerperSearchTool(call_name="search", num_results=5)

    @parameterized.expand([("empty", ""), ("whitespace", "   \n\t  "), ("none", None)])
    def test_empty_query_returns_error(self, name, query_input):
        """Test that empty/whitespace/None query returns an error without calling API."""
        result = asyncio.run(self.tool.execute(query_input))

        expected = ToolOutput(
            output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
        )
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            (
                "organic_results",
                {
                    "organic": [
                        {"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"},
                        {"title": "Result 2", "snippet": "This is result 2", "link": "https://example.com/2"},
                    ]
                },
                ["Result 1", "This is result 1", "https://example.com/1", "Result 2"],
            ),
            (
                "answer_box",
                {
                    "answerBox": {"answer": "The answer is 42"},
                    "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
                },
                ["Direct Answer", "The answer is 42", "Result 1"],
            ),
            (
                "featured_snippet",
                {
                    "answerBox": {"snippet": "This is a featured snippet"},
                    "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
                },
                ["Featured Snippet", "This is a featured snippet"],
            ),
        ]
    )
    @patch("open_instruct.tools.tools.make_api_request")
    def test_successful_search(self, name, api_data, expected_in_output, mock_api_request):
        """Test successful search with various response types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data=api_data)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("test query"))

        self.assertIsInstance(result, ToolOutput)
        for expected in expected_in_output:
            self.assertIn(expected, result.output)
        self.assertEqual(result.error, "")
        self.assertTrue(result.called)
        self.assertFalse(result.timeout)

    @parameterized.expand(
        [
            ("timeout", {"error": "Timeout after 10 seconds", "timed_out": True}, True, "Timeout after 10 seconds"),
            (
                "connection_error",
                {"error": "Connection error: Connection refused", "timed_out": False},
                False,
                "Connection error",
            ),
            ("http_error", {"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ]
    )
    @patch("open_instruct.tools.tools.make_api_request")
    def test_error_handling(self, name, api_response, expected_timeout, expected_error_contains, mock_api_request):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("test query"))

        self.assertTrue(result.called)
        self.assertEqual(result.timeout, expected_timeout)
        self.assertIn(expected_error_contains, result.error)
        self.assertIn(expected_error_contains, result.output)  # Error message also in output for model feedback

    @patch("open_instruct.tools.tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"organic": []})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("nonexistent query"))

        self.assertTrue(result.called)
        self.assertEqual(result.error, "Query returned no results.")
        self.assertEqual(result.output, result.error)  # Error in output for model feedback
        self.assertFalse(result.timeout)

    @patch("open_instruct.tools.tools.make_api_request")
    def test_results_without_snippets_filtered_out(self, mock_api_request):
        """Test that results without snippets are filtered out."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(
                data={
                    "organic": [
                        {"title": "Result 1", "link": "https://example.com/1"},  # No snippet
                        {"title": "Result 2", "snippet": "This has a snippet", "link": "https://example.com/2"},
                    ]
                }
            )

        mock_api_request.side_effect = mock_response

        result = asyncio.run(self.tool.execute("test query"))

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

    def test_tool_class_attribute(self):
        """Test tool_class is set to SerperSearchTool."""
        self.assertEqual(SerperSearchToolConfig.tool_class, SerperSearchTool)


class TestToolsConfig(unittest.TestCase):
    """Tests for ToolsConfig dataclass."""

    def test_no_tools_configured(self):
        """Test ToolsConfig works when no tools are configured."""
        config = ToolsConfig()
        self.assertEqual(config.tools, [])
        self.assertEqual(config.tool_configs, [])
        self.assertEqual(config._parsed_tools, [])
        self.assertEqual(config.tool_call_names, [])
        self.assertFalse(config.enabled)

    def test_tools_with_default_configs(self):
        """Test ToolsConfig sets default tool_configs when not provided."""
        config = ToolsConfig(tools=["python", "search"])
        self.assertEqual(config.tools, ["python", "search"])
        self.assertEqual(config.tool_configs, ["{}", "{}"])
        self.assertEqual(config._parsed_tools, [
            ParsedToolConfig(name="python", call_name="python", config={}),
            ParsedToolConfig(name="search", call_name="search", config={}),
        ])
        self.assertEqual(config.tool_call_names, ["python", "search"])
        self.assertTrue(config.enabled)

    def test_tools_with_custom_configs(self):
        """Test ToolsConfig parses custom tool_configs from JSON strings."""
        config = ToolsConfig(tools=["python", "search"], tool_configs=['{"timeout": 10}', '{"num_results": 5}'])
        self.assertEqual(config.tool_configs, ['{"timeout": 10}', '{"num_results": 5}'])
        self.assertEqual(config._parsed_tools, [
            ParsedToolConfig(name="python", call_name="python", config={"timeout": 10}),
            ParsedToolConfig(name="search", call_name="search", config={"num_results": 5}),
        ])

    def test_tools_with_custom_call_names(self):
        """Test ToolsConfig allows custom tool_call_names."""
        config = ToolsConfig(tools=["python", "search"], tool_call_names=["code", "web_search"])
        self.assertEqual(config.tool_call_names, ["code", "web_search"])
        self.assertEqual(config._parsed_tools[0].call_name, "code")
        self.assertEqual(config._parsed_tools[1].call_name, "web_search")

    def test_mismatched_tool_configs_length_raises(self):
        """Test ToolsConfig raises when tool_configs length doesn't match tools."""
        with self.assertRaisesRegex(ValueError, "tool_configs must have same length as tools"):
            ToolsConfig(tools=["python", "search"], tool_configs=["{}"])

    def test_mismatched_tool_call_names_length_raises(self):
        """Test ToolsConfig raises when tool_call_names length doesn't match tools."""
        with self.assertRaisesRegex(ValueError, "tool_call_names must have same length as tools"):
            ToolsConfig(tools=["python", "search"], tool_call_names=["code"])

    def test_invalid_tool_config_json_raises(self):
        """Test ToolsConfig raises on invalid JSON in tool_configs."""
        with self.assertRaisesRegex(ValueError, "Invalid tool_config for tool python"):
            ToolsConfig(tools=["python"], tool_configs=["not valid json"])


if __name__ == "__main__":
    unittest.main()
