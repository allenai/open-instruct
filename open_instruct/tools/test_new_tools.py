"""Tests for new tools (PythonCodeTool, S2SearchTool, and SerperSearchTool from new_tools.py)."""

import asyncio
import dataclasses
from unittest.mock import patch

import pytest

from open_instruct.tools.new_tools import (
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
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

    def test_get_openai_tool_definitions(self):
        """Test OpenAI tool definition format."""
        tool = PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute")
        definition = get_openai_tool_definitions(tool)

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "python"


class TestPythonCodeToolExecution:
    """Tests for PythonCodeTool execution (async execute)."""

    @pytest.fixture
    def tool(self):
        """Set up test fixture."""
        return PythonCodeTool(call_name="python", api_endpoint="http://localhost:1212/execute", timeout=3)

    @pytest.mark.parametrize("code_input", ["", "   \n\t  ", None], ids=["empty", "whitespace", "none"])
    def test_empty_code_returns_error(self, tool, code_input):
        """Test that empty/whitespace/None code returns an error without calling API."""
        result = asyncio.run(tool(code_input))

        assert isinstance(result, ToolOutput)
        assert result.output == ""
        assert result.error == "Empty code. Please provide some code to execute."
        assert result.called

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_execution(self, mock_api_request, tool):
        """Test successful code execution."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"output": "Hello, World!\n", "error": None})

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool("print('Hello, World!')"))

        assert result.output == "Hello, World!\n"
        assert result.error == ""
        assert result.called

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_timeout_handling(self, mock_api_request, tool):
        """Test timeout is handled correctly."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(error="Timeout after 3 seconds", timed_out=True)

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool("import time; time.sleep(100)"))

        assert result.timeout
        assert "Timeout" in result.output


class TestPythonCodeToolConfig:
    """Tests for PythonCodeToolConfig."""

    def test_config_requires_api_endpoint(self):
        """Test config requires api_endpoint."""
        fields = {f.name: f for f in dataclasses.fields(PythonCodeToolConfig)}
        assert "api_endpoint" in fields

    def test_tool_class_attribute(self):
        """Test tool_class is set to PythonCodeTool."""
        assert PythonCodeToolConfig.tool_class == PythonCodeTool


class TestTruncateHelper:
    """Tests for _truncate helper function."""

    @pytest.mark.parametrize(
        "text,max_length,should_truncate",
        [("Hello, World!", 500, False), ("a" * 600, 500, True)],
        ids=["short_text", "long_text"],
    )
    def test_truncate(self, text, max_length, should_truncate):
        """Test _truncate with various inputs."""
        result = _truncate(text, max_length=max_length)
        if should_truncate:
            assert "more chars" in result
        else:
            assert result == text


class TestS2SearchToolInit:
    """Tests for S2SearchTool initialization and properties."""

    @pytest.mark.parametrize(
        "num_results,expected_num_results", [(None, 10), (5, 5)], ids=["defaults", "custom_values"]
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

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing S2_API_KEY raises a ValueError on initialization."""
        with pytest.raises(ValueError, match="Missing S2_API_KEY"):
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

        assert result.output == ""
        assert result.error == "Empty query. Please provide a search query."
        assert result.called is True

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

        assert "First research paper snippet" in result.output
        assert "Second research paper snippet" in result.output
        assert result.error == ""
        assert result.called is True

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request, tool):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"data": []})

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool.execute("nonexistent query"))

        assert result.error == "Query returned no results."
        assert result.output == result.error  # Error in output for model feedback

    @pytest.mark.parametrize(
        "api_response,expected_timeout,expected_error_contains",
        [
            ({"error": "Timeout after 60 seconds", "timed_out": True}, True, "Timeout"),
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
        assert expected_error_contains in result.output

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

    def test_tool_class_attribute(self):
        """Test tool_class is set to S2SearchTool."""
        assert S2SearchToolConfig.tool_class == S2SearchTool


class TestSerperSearchToolInit:
    """Tests for SerperSearchTool initialization and properties."""

    @pytest.mark.parametrize(
        "num_results,expected_num_results", [(None, 5), (10, 10)], ids=["defaults", "custom_values"]
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

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_error_on_init(self):
        """Test that missing SERPER_API_KEY raises a ValueError on initialization."""
        with pytest.raises(ValueError, match="Missing SERPER_API_KEY"):
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

        assert result.output == ""
        assert result.error == "Empty query. Please provide a search query."
        assert result.called is True

    @pytest.mark.parametrize(
        "api_data,expected_in_output",
        [
            (
                {"organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}]},
                ["Result 1", "This is result 1"],
            ),
            (
                {
                    "answerBox": {"answer": "The answer is 42"},
                    "organic": [{"title": "Result 1", "snippet": "This is result 1", "link": "https://example.com/1"}],
                },
                ["Direct Answer", "The answer is 42"],
            ),
        ],
        ids=["organic_results", "answer_box"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_successful_search(self, mock_api_request, tool, api_data, expected_in_output):
        """Test successful search with various response types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data=api_data)

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool.execute("test query"))

        for expected in expected_in_output:
            assert expected in result.output
        assert result.error == ""

    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_no_results_returns_error(self, mock_api_request, tool):
        """Test query with no results returns an error."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(data={"organic": []})

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool.execute("nonexistent query"))

        assert result.error == "Query returned no results."
        assert result.output == result.error  # Error in output for model feedback

    @pytest.mark.parametrize(
        "api_response,expected_timeout,expected_error_contains",
        [
            ({"error": "Timeout after 10 seconds", "timed_out": True}, True, "Timeout"),
            ({"error": "Connection error: Connection refused", "timed_out": False}, False, "Connection error"),
        ],
        ids=["timeout", "connection_error"],
    )
    @patch("open_instruct.tools.new_tools.make_api_request")
    def test_error_handling(self, mock_api_request, tool, api_response, expected_timeout, expected_error_contains):
        """Test error handling for various API error types."""
        from open_instruct.tools.utils import APIResponse

        async def mock_response(*args, **kwargs):
            return APIResponse(**api_response)

        mock_api_request.side_effect = mock_response
        result = asyncio.run(tool.execute("test query"))

        assert result.timeout is expected_timeout
        assert expected_error_contains in result.error
        assert expected_error_contains in result.output


class TestSerperSearchToolConfig:
    """Tests for SerperSearchToolConfig."""

    def test_config_default_values(self):
        """Test config has correct default values."""
        config = SerperSearchToolConfig()
        assert config.num_results == 5

    def test_tool_class_attribute(self):
        """Test tool_class is set to SerperSearchTool."""
        assert SerperSearchToolConfig.tool_class == SerperSearchTool


if __name__ == "__main__":
    pytest.main([__file__])
