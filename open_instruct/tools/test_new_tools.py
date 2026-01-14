"""Tests for new tools (PythonCodeTool and SerperSearchTool from new_tools.py)."""

import asyncio
import dataclasses
from unittest.mock import patch

import pytest

from open_instruct.tools.new_tools import (
    PythonCodeTool,
    PythonCodeToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
    _truncate,
)
from open_instruct.tools.utils import ToolOutput, get_openai_tool_definitions


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
        assert result.output == ""
        assert result.error == "Query returned no results."
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


if __name__ == "__main__":
    pytest.main([__file__])
