"""Tests for new tools (PythonCodeTool, JinaBrowseTool, S2SearchTool, and SerperSearchTool from new_tools.py)."""

import asyncio
import dataclasses
from unittest.mock import patch

import pytest

from open_instruct.tools.new_tools import (
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
from open_instruct.tools.utils import ToolOutput, ToolsConfig, get_openai_tool_definitions


class TestPythonCodeToolInit:
    """Tests for PythonCodeTool initialization and properties."""

    @pytest.mark.parametrize(
        "call_name,api_endpoint,timeout,expected_timeout",
        [
            ("python", "http://localhost:1212/execute", None, 3),  # default timeout
            ("code", "http://example.com/run", 10, 10),  # custom values
        ],
        ids=["defaults", "custom_values"],
    )
    def test_initialization(self, call_name, api_endpoint, timeout, expected_timeout):
        """Test tool initializes with correct values."""
        tool = (
            PythonCodeTool(call_name=call_name, api_endpoint=api_endpoint)
            if timeout is None
            else PythonCodeTool(call_name=call_name, api_endpoint=api_endpoint, timeout=timeout)
        )

        assert tool.api_endpoint == api_endpoint
        assert tool.timeout == expected_timeout
        assert tool.call_name == call_name

    @pytest.mark.parametrize("call_name", ["python", "code"])
    def test_tool_call_name(self, call_name):
        """Test tool call_name can be customized."""
        tool = PythonCodeTool(call_name=call_name, api_endpoint="http://localhost:1212/execute")
        assert tool.call_name == call_name

    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        assert tool.description == "Executes Python code and returns printed output."

    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        params = tool.parameters

        assert params["type"] == "object"
        assert "code" in params["properties"]
        assert "code" in params["required"]
        assert params["properties"]["code"]["description"] == "Python code to execute"

    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        definition = get_openai_tool_definitions(tool)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "python"
        assert definition["function"]["description"] == "Executes Python code and returns printed output."
        assert "parameters" in definition["function"]

    @pytest.mark.parametrize(
        "custom_call_name,expected_call_name",
        [
            ("code", "code"),
            (None, "python"),  # default uses config_name
        ],
        ids=["custom_name", "default_name"],
    )
    def test_call_name_via_config(self, custom_call_name, expected_call_name):
        """Test that call_name can be set when instantiating tools from config."""
        config = PythonCodeToolConfig(api_endpoint="http://example.com/execute")
        # Instantiate tool directly with config values
        call_name = custom_call_name if custom_call_name else config.tool_class.config_name
        tool = PythonCodeTool(call_name=call_name, api_endpoint=config.api_endpoint, timeout=config.timeout)
        assert tool.call_name == expected_call_name

        # Verify OpenAI definition uses the correct name
        definition = get_openai_tool_definitions(tool)
        assert definition["function"]["name"] == expected_call_name


class TestPythonCodeToolExecution:
    """Tests for PythonCodeTool execution (async execute)."""

    @pytest.fixture
    def tool(self):
        """Set up test fixture."""
        return PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute", timeout=3)

    @pytest.mark.parametrize(
        "code_input",
        [
            "",  # empty string
            "   \n\t  ",  # whitespace only
            None,  # None value
        ],
        ids=["empty", "whitespace", "none"],
    )
    def test_empty_code_returns_error(self, tool, code_input):
        """Test that empty/whitespace/None code returns an error without calling API."""
        result = asyncio.run(tool(code_input))

        assert isinstance(result, ToolOutput)
        assert result.output == ""
        assert result.error == "Empty code. Please provide some code to execute."
        assert result.called
        assert not result.timeout
        assert result.runtime == 0

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_execution(self, mock_api_request, tool):
        """Test successful code execution."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "Hello, World!\n", "error": None})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool("print('Hello, World!')"))

        assert isinstance(result, ToolOutput)
        assert result.output == "Hello, World!\n"
        assert result.error == ""
        assert result.called
        assert not result.timeout
        assert result.runtime > 0

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_execution_with_error_response(self, mock_api_request, tool):
        """Test code execution that returns an error from the API."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "", "error": "NameError: name 'undefined_var' is not defined"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool("print(undefined_var)"))

        assert result.called
        assert "NameError" in result.output
        assert result.error == "NameError: name 'undefined_var' is not defined"
        assert not result.timeout

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_timeout_handling(self, mock_api_request, tool):
        """Test timeout is handled correctly."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(error="Timeout after 3 seconds", timed_out=True)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool("import time; time.sleep(100)"))

        assert result.called
        assert result.timeout
        assert "Timeout after 3 seconds" in result.output

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_api_connection_error(self, mock_api_request, tool):
        """Test handling of API connection errors."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(error="Connection error: Connection refused")

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool("print('test')"))

        assert result.called
        assert not result.timeout
        assert "Connection error" in result.output
        assert "Connection refused" in result.output

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_execution_with_output_and_error(self, mock_api_request, tool):
        """Test execution that produces both output and an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "Partial output\n", "error": "Some warning or error"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool("print('Partial output'); raise Exception()"))

        assert result.called
        assert "Partial output" in result.output
        assert "Some warning or error" in result.output
        assert result.error == "Some warning or error"

    @patch("open_instruct.tools.new_tools.make_api_request")
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
        assert call_args[1]["timeout_seconds"] == 10


class TestPythonCodeToolConfig:
    """Tests for PythonCodeToolConfig."""

    def test_config_requires_api_endpoint(self):
        """Test config requires api_endpoint."""
        fields = {f.name: f for f in dataclasses.fields(PythonCodeToolConfig)}
        assert "api_endpoint" in fields
        assert fields["api_endpoint"].default == dataclasses.MISSING
        assert fields["api_endpoint"].default_factory == dataclasses.MISSING

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = PythonCodeToolConfig(api_endpoint="http://example.com/execute", timeout=5)

        assert config.api_endpoint == "http://example.com/execute"
        assert config.timeout == 5

    def test_config_values_used_for_tool(self):
        """Test config values can be used to create tool correctly."""
        config = PythonCodeToolConfig(api_endpoint="http://localhost:1212/execute", timeout=5)

        # Create tool using config values (as build_remote does internally)
        tool = PythonCodeTool(
            call_name=config.tool_class.config_name, api_endpoint=config.api_endpoint, timeout=config.timeout
        )

        assert isinstance(tool, PythonCodeTool)
        assert tool.api_endpoint == "http://localhost:1212/execute"
        assert tool.timeout == 5

    def test_tool_class_attribute(self):
        """Test tool_class is set to PythonCodeTool."""
        assert PythonCodeToolConfig.tool_class == PythonCodeTool


class TestTruncateHelper:
    """Tests for _truncate helper function."""

    @pytest.mark.parametrize(
        "text,max_length,expected_result,should_truncate",
        [
            ("Hello, World!", 500, "Hello, World!", False),  # short text
            ("a" * 600, 500, "a" * 500 + "... [100 more chars]", True),  # long text
            ("a" * 500, 500, "a" * 500, False),  # exact boundary
            ("Hello, World! This is a test.", 10, "Hello, Wor", True),  # custom max_length
        ],
        ids=["short_text", "long_text", "exact_boundary", "custom_max_length"],
    )
    def test_truncate(self, text, max_length, expected_result, should_truncate):
        """Test _truncate with various inputs."""
        result = _truncate(text, max_length=max_length)

        if should_truncate:
            assert result.startswith(text[:max_length])
            assert "more chars" in result
            if text == "a" * 600:
                assert len(result) == 500 + len("... [100 more chars]")
        else:
            assert result == expected_result


class TestJinaBrowseToolInit:
    """Tests for JinaBrowseTool initialization and properties."""

    @pytest.mark.parametrize(
        "timeout,expected_timeout",
        [
            (None, 30),  # default
            (60, 60),  # custom
        ],
        ids=["defaults", "custom_values"],
    )
    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_initialization(self, timeout, expected_timeout):
        """Test tool initializes with correct values."""
        tool = (
            JinaBrowseTool(call_name="browse")
            if timeout is None
            else JinaBrowseTool(call_name="browse", timeout=timeout)
        )

        assert tool.timeout == expected_timeout
        assert tool.call_name == "browse"
        assert tool.config_name == "jina_browse"
        assert tool.api_key == "test_key"

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = JinaBrowseTool(call_name="browse")
        assert tool.description == "Fetches and converts webpage content to clean markdown using Jina Reader API"

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = JinaBrowseTool(call_name="browse")
        params = tool.parameters

        assert params["type"] == "object"
        assert "url" in params["properties"]
        assert "url" in params["required"]
        assert params["properties"]["url"]["description"] == "The URL of the webpage to fetch"

    @patch.dict("os.environ", {"JINA_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = JinaBrowseTool(call_name="browse")
        definition = get_openai_tool_definitions(tool)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "browse"
        assert "parameters" in definition["function"]

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing JINA_API_KEY raises a ValueError on initialization."""
        with pytest.raises(ValueError, match="Missing JINA_API_KEY environment variable."):
            JinaBrowseTool(call_name="browse")


class TestJinaBrowseToolExecution:
    """Tests for JinaBrowseTool execution (async execute)."""

    @pytest.fixture
    def tool(self):
        """Set up test fixture."""
        with patch.dict("os.environ", {"JINA_API_KEY": "test_key"}):
            return JinaBrowseTool(call_name="browse", timeout=30)

    @pytest.mark.parametrize("url_input", ["", "   \n\t  ", None], ids=["empty", "whitespace", "none"])
    def test_empty_url_returns_error(self, tool, url_input):
        """Test that empty/whitespace/None URL returns an error without calling API."""
        result = asyncio.run(tool.execute(url_input))

        assert isinstance(result, ToolOutput)
        assert result.output == ""
        assert result.error == "Empty URL. Please provide a URL to fetch."
        assert result.called is True
        assert result.timeout is False

    @pytest.mark.parametrize(
        "api_data,expected_in_output",
        [
            # With title
            (
                {"code": 200, "data": {"title": "Example Page", "content": "This is the page content."}},
                ["# Example Page", "This is the page content."],
            ),
            # Without title
            ({"code": 200, "data": {"content": "Just content without a title."}}, ["Just content without a title."]),
        ],
        ids=["with_title", "without_title"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_fetch(self, mock_api_request, tool, api_data, expected_in_output):
        """Test successful webpage fetch with various response types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data=api_data)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("https://example.com"))

        assert isinstance(result, ToolOutput)
        for expected in expected_in_output:
            assert expected in result.output
        assert result.error == ""
        assert result.called is True
        assert result.timeout is False

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_jina_api_error_response(self, mock_api_request, tool):
        """Test handling of Jina API error response (code != 200)."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"code": 400, "message": "Invalid URL format"})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("https://invalid-url"))

        assert result.called is True
        assert "Jina API error" in result.error
        assert "Invalid URL format" in result.error
        assert "Jina API error" in result.output  # Error in output for model feedback

    @pytest.mark.parametrize(
        "api_response,expected_timeout,expected_error_contains",
        [
            ({"error": "Timeout after 30 seconds", "timed_out": True}, True, "Timeout after 30 seconds"),
            ({"error": "Connection error: Connection refused", "timed_out": False}, False, "Connection error"),
            ({"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ],
        ids=["timeout", "connection_error", "http_error"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_error_handling(self, mock_api_request, tool, api_response, expected_timeout, expected_error_contains):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("https://example.com"))

        assert result.called is True
        assert result.timeout is expected_timeout
        assert expected_error_contains in result.error
        assert expected_error_contains in result.output  # Error message also in output for model feedback

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_uses_get_method(self, mock_api_request, tool):
        """Test that JinaBrowseTool uses GET method."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"code": 200, "data": {"content": "OK"}})

        mock_api_request.side_effect = mock_response

        asyncio.run(tool.execute("https://example.com"))

        mock_api_request.assert_called_once()
        call_args = mock_api_request.call_args
        assert call_args[1]["method"] == "GET"
        assert "r.jina.ai" in call_args[1]["url"]


class TestJinaBrowseToolConfig:
    """Tests for JinaBrowseToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = JinaBrowseToolConfig()
        assert config.timeout == 30

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = JinaBrowseToolConfig(timeout=60)
        assert config.timeout == 60

    def test_tool_class_attribute(self):
        """Test tool_class is set to JinaBrowseTool."""
        assert JinaBrowseToolConfig.tool_class == JinaBrowseTool


class TestS2SearchToolInit:
    """Tests for S2SearchTool initialization and properties."""

    @pytest.mark.parametrize(
        "num_results,expected_num_results",
        [
            (None, 10),  # default
            (5, 5),  # custom
        ],
        ids=["defaults", "custom_values"],
    )
    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_initialization(self, num_results, expected_num_results):
        """Test tool initializes with correct values."""
        tool = (
            S2SearchTool(call_name="s2")
            if num_results is None
            else S2SearchTool(call_name="s2", num_results=num_results)
        )

        assert tool.num_results == expected_num_results
        assert tool.call_name == "s2"
        assert tool.config_name == "s2_search"

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = S2SearchTool(call_name="s2")
        assert "Semantic Scholar" in tool.description

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = S2SearchTool(call_name="s2")
        params = tool.parameters

        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "query" in params["required"]

    @patch.dict("os.environ", {"S2_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = S2SearchTool(call_name="s2")
        definition = get_openai_tool_definitions(tool)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "s2"
        assert "parameters" in definition["function"]

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing S2_API_KEY raises a ValueError on initialization."""
        with pytest.raises(ValueError, match="Missing S2_API_KEY environment variable."):
            S2SearchTool(call_name="s2")


class TestS2SearchToolExecution:
    """Tests for S2SearchTool execution (async execute)."""

    @pytest.fixture
    def tool(self):
        """Set up test fixture."""
        with patch.dict("os.environ", {"S2_API_KEY": "test_key"}):
            return S2SearchTool(call_name="s2", num_results=10)

    @pytest.mark.parametrize("query_input", ["", "   \n\t  ", None], ids=["empty", "whitespace", "none"])
    def test_empty_query_returns_error(self, tool, query_input):
        """Test that empty/whitespace/None query returns an error without calling API."""
        result = asyncio.run(tool.execute(query_input))

        assert isinstance(result, ToolOutput)
        assert result.output == ""
        assert result.error == "Empty query. Please provide a search query."
        assert result.called is True
        assert result.timeout is False

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_search(self, mock_api_request, tool):
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

        result = asyncio.run(tool.execute("machine learning"))

        assert isinstance(result, ToolOutput)
        assert "First research paper snippet" in result.output
        assert "Second research paper snippet" in result.output
        assert result.error == ""
        assert result.called is True
        assert result.timeout is False

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request, tool):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"data": []})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("nonexistent query"))

        assert result.called is True
        assert result.error == "Query returned no results."
        assert result.output == result.error  # Error in output for model feedback
        assert result.timeout is False

    @pytest.mark.parametrize(
        "api_response,expected_timeout,expected_error_contains",
        [
            ({"error": "Timeout after 60 seconds", "timed_out": True}, True, "Timeout after 60 seconds"),
            ({"error": "Connection error: Connection refused", "timed_out": False}, False, "Connection error"),
            ({"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ],
        ids=["timeout", "connection_error", "http_error"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_error_handling(self, mock_api_request, tool, api_response, expected_timeout, expected_error_contains):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("test query"))

        assert result.called is True
        assert result.timeout is expected_timeout
        assert expected_error_contains in result.error
        assert expected_error_contains in result.output  # Error message also in output for model feedback

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_uses_get_method(self, mock_api_request, tool):
        """Test that S2SearchTool uses GET method."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"data": []})

        mock_api_request.side_effect = mock_response

        asyncio.run(tool.execute("test query"))

        mock_api_request.assert_called_once()
        call_args = mock_api_request.call_args
        assert call_args[1]["method"] == "GET"


class TestS2SearchToolConfig:
    """Tests for S2SearchToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = S2SearchToolConfig()
        assert config.num_results == 10
        assert config.timeout == 60

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = S2SearchToolConfig(num_results=5, timeout=30)
        assert config.num_results == 5
        assert config.timeout == 30

    def test_tool_class_attribute(self):
        """Test tool_class is set to S2SearchTool."""
        assert S2SearchToolConfig.tool_class == S2SearchTool


class TestSerperSearchToolInit:
    """Tests for SerperSearchTool initialization and properties."""

    @pytest.mark.parametrize(
        "num_results,expected_num_results",
        [
            (None, 5),  # default
            (10, 10),  # custom
        ],
        ids=["defaults", "custom_values"],
    )
    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_initialization(self, num_results, expected_num_results):
        """Test tool initializes with correct values."""
        tool = (
            SerperSearchTool(call_name="search")
            if num_results is None
            else SerperSearchTool(call_name="search", num_results=num_results)
        )

        assert tool.num_results == expected_num_results
        assert tool.call_name == "search"
        assert tool.config_name == "serper_search"

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_description(self):
        """Test tool description is set correctly."""
        tool = SerperSearchTool(call_name="search")
        assert tool.description == "Google search via the Serper API"

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_tool_parameters_schema(self):
        """Test tool parameters schema is correct."""
        tool = SerperSearchTool(call_name="search")
        params = tool.parameters

        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "query" in params["required"]
        assert params["properties"]["query"]["description"] == "The search query for Google"

    @patch.dict("os.environ", {"SERPER_API_KEY": "test_key"})
    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = SerperSearchTool(call_name="search")
        definition = get_openai_tool_definitions(tool)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search"
        assert definition["function"]["description"] == "Google search via the Serper API"
        assert "parameters" in definition["function"]

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing SERPER_API_KEY raises a ValueError on initialization."""
        with pytest.raises(ValueError, match="Missing SERPER_API_KEY environment variable."):
            SerperSearchTool(call_name="search")


class TestSerperSearchToolExecution:
    """Tests for SerperSearchTool execution (async execute)."""

    @pytest.fixture
    def tool(self):
        """Set up test fixture."""
        with patch.dict("os.environ", {"SERPER_API_KEY": "test_key"}):
            return SerperSearchTool(call_name="search", num_results=5)

    @pytest.mark.parametrize("query_input", ["", "   \n\t  ", None], ids=["empty", "whitespace", "none"])
    def test_empty_query_returns_error(self, tool, query_input):
        """Test that empty/whitespace/None query returns an error without calling API."""
        result = asyncio.run(tool.execute(query_input))

        assert isinstance(result, ToolOutput)
        assert result.output == ""
        assert result.error == "Empty query. Please provide a search query."
        assert result.called is True
        assert result.timeout is False

    @pytest.mark.parametrize(
        "api_data,expected_in_output",
        [
            # Organic results
            (
                {
                    "organic": [
                        {"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"},
                        {"title": "Result 2", "snippet": "This is result 2", "link": "https://example.com/2"},
                    ]
                },
                ["Result 1", "This is result 1", "https://example.com/1", "Result 2"],
            ),
            # Answer box
            (
                {
                    "answerBox": {"answer": "The answer is 42"},
                    "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
                },
                ["Direct Answer", "The answer is 42", "Result 1"],
            ),
            # Featured snippet
            (
                {
                    "answerBox": {"snippet": "This is a featured snippet"},
                    "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
                },
                ["Featured Snippet", "This is a featured snippet"],
            ),
        ],
        ids=["organic_results", "answer_box", "featured_snippet"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_search(self, mock_api_request, tool, api_data, expected_in_output):
        """Test successful search with various response types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data=api_data)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("test query"))

        assert isinstance(result, ToolOutput)
        for expected in expected_in_output:
            assert expected in result.output
        assert result.error == ""
        assert result.called is True
        assert result.timeout is False

    @pytest.mark.parametrize(
        "api_response,expected_timeout,expected_error_contains",
        [
            ({"error": "Timeout after 10 seconds", "timed_out": True}, True, "Timeout after 10 seconds"),
            ({"error": "Connection error: Connection refused", "timed_out": False}, False, "Connection error"),
            ({"error": "HTTP error: 403 Forbidden", "timed_out": False}, False, "HTTP error"),
        ],
        ids=["timeout", "connection_error", "http_error"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_error_handling(self, mock_api_request, tool, api_response, expected_timeout, expected_error_contains):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("test query"))

        assert result.called is True
        assert result.timeout is expected_timeout
        assert expected_error_contains in result.error
        assert expected_error_contains in result.output  # Error message also in output for model feedback

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request, tool):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"organic": []})

        mock_api_request.side_effect = mock_response

        result = asyncio.run(tool.execute("nonexistent query"))

        assert result.called is True
        assert result.error == "Query returned no results."
        assert result.output == result.error  # Error in output for model feedback
        assert result.timeout is False

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_results_without_snippets_filtered_out(self, mock_api_request, tool):
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

        result = asyncio.run(tool.execute("test query"))

        assert "Result 1" not in result.output
        assert "Result 2" in result.output
        assert "This has a snippet" in result.output


class TestSerperSearchToolConfig:
    """Tests for SerperSearchToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = SerperSearchToolConfig()
        assert config.num_results == 5

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = SerperSearchToolConfig(num_results=10)
        assert config.num_results == 10

    def test_tool_class_attribute(self):
        """Test tool_class is set to SerperSearchTool."""
        assert SerperSearchToolConfig.tool_class == SerperSearchTool


class TestToolsConfig:
    """Tests for ToolsConfig dataclass."""

    def test_no_tools_configured(self):
        """Test ToolsConfig works when no tools are configured."""
        config = ToolsConfig()
        assert config.tools is None
        assert config.tool_configs == []
        assert config._parsed_tool_configs == []
        assert config.tool_call_names is None
        assert config.enabled is False

    def test_tools_with_default_configs(self):
        """Test ToolsConfig sets default tool_configs when not provided."""
        config = ToolsConfig(tools=["python", "search"])
        assert config.tools == ["python", "search"]
        assert config.tool_configs == ["{}", "{}"]
        assert config._parsed_tool_configs == [{}, {}]
        assert config.tool_call_names == ["python", "search"]
        assert config.enabled is True

    def test_tools_with_custom_configs(self):
        """Test ToolsConfig parses custom tool_configs from JSON strings."""
        config = ToolsConfig(tools=["python", "search"], tool_configs=['{"timeout": 10}', '{"num_results": 5}'])
        assert config.tool_configs == ['{"timeout": 10}', '{"num_results": 5}']
        assert config._parsed_tool_configs == [{"timeout": 10}, {"num_results": 5}]

    def test_tools_with_custom_call_names(self):
        """Test ToolsConfig allows custom tool_call_names."""
        config = ToolsConfig(tools=["python", "search"], tool_call_names=["code", "web_search"])
        assert config.tool_call_names == ["code", "web_search"]

    def test_mismatched_tool_configs_length_raises(self):
        """Test ToolsConfig raises when tool_configs length doesn't match tools."""
        with pytest.raises(ValueError, match="tool_configs must have same length as tools"):
            ToolsConfig(tools=["python", "search"], tool_configs=["{}"])

    def test_mismatched_tool_call_names_length_raises(self):
        """Test ToolsConfig raises when tool_call_names length doesn't match tools."""
        with pytest.raises(ValueError, match="tool_call_names must have same length as tools"):
            ToolsConfig(tools=["python", "search"], tool_call_names=["code"])

    def test_invalid_tool_config_json_raises(self):
        """Test ToolsConfig raises on invalid JSON in tool_configs."""
        with pytest.raises(ValueError, match="Invalid tool_config for tool python"):
            ToolsConfig(tools=["python"], tool_configs=["not valid json"])


if __name__ == "__main__":
    pytest.main([__file__])