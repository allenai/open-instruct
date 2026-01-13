"""Tests for tool parsers."""

import unittest
from unittest.mock import MagicMock, patch

from open_instruct.tools.parsers import OpenInstructLegacyToolParser, get_available_parsers


class MockTool:
    """Mock tool for testing without ray."""

    def __init__(self, name: str, param_name: str = "text", required: list[str] | None = None):
        self.call_name = name
        self.param_name = param_name
        self.required = required if required is not None else [param_name]

    def get_call_name(self):
        return self.call_name

    def get_parameters(self):
        return {"required": self.required, "properties": {self.param_name: {"type": "string"}}}


def create_mock_tool_actor(name: str, param_name: str = "text", required: list[str] | None = None) -> MagicMock:
    """Create a mock tool actor handle that works with ray.get()."""
    mock_tool = MockTool(name, param_name, required)
    actor_handle = MagicMock()
    actor_handle.get_call_name.remote.return_value = mock_tool.get_call_name()
    actor_handle.get_parameters.remote.return_value = mock_tool.get_parameters()
    return actor_handle


class TestOpenInstructLegacyToolParser(unittest.TestCase):
    """Tests for OpenInstructLegacyToolParser."""

    def setUp(self):
        """Set up mock actors for each test."""
        self.patcher = patch("open_instruct.tools.parsers.ray")
        self.mock_ray = self.patcher.start()
        # Make ray.get return the value directly (simulating sync behavior)
        self.mock_ray.get.side_effect = lambda x: x

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_single_tool_extraction(self):
        """Test extracting a single tool call."""
        mock_actor = create_mock_tool_actor("search", param_name="query")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "I need to search for something. <search>python tutorials</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "search")
        self.assertEqual(tool_calls[0].args, {"query": "python tutorials"})

    def test_multiple_tools_extraction(self):
        """Test extracting multiple different tool calls."""
        mock_search = create_mock_tool_actor("search", param_name="query")
        mock_code = create_mock_tool_actor("code", param_name="script")
        parser = OpenInstructLegacyToolParser([mock_search, mock_code])

        text = "First search <search>python</search> then run <code>print('hello')</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        tool_names = {tc.name for tc in tool_calls}
        self.assertEqual(tool_names, {"search", "code"})

    def test_no_tool_calls(self):
        """Test that no tool calls are returned when none exist."""
        mock_actor = create_mock_tool_actor("search")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "This is just regular text without any tool calls."
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_multiline_content(self):
        """Test extracting tool calls with multiline content."""
        mock_actor = create_mock_tool_actor("code", param_name="script")
        parser = OpenInstructLegacyToolParser([mock_actor])

        code_content = """def hello():
    print('Hello, World!')

hello()"""
        text = f"Here's some code:\n<code>{code_content}</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "code")
        self.assertEqual(tool_calls[0].args["script"], code_content)

    def test_partial_tag_not_matched(self):
        """Test that incomplete tags are not matched."""
        mock_actor = create_mock_tool_actor("search")
        parser = OpenInstructLegacyToolParser([mock_actor])

        # Missing closing tag
        text = "Here's a search <search>query without closing"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_nested_content_with_angle_brackets(self):
        """Test content containing angle brackets."""
        mock_actor = create_mock_tool_actor("code", param_name="script")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "<code>if x > 5 and y < 10: print('yes')</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["script"], "if x > 5 and y < 10: print('yes')")

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
        mock_actor = create_mock_tool_actor("search")
        parser = OpenInstructLegacyToolParser([mock_actor])

        result = parser.format_tool_outputs(["Search result: Found 5 items"])
        expected = "<output>\nSearch result: Found 5 items\n</output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_multiple(self):
        """Test formatting multiple tool outputs."""
        mock_actor = create_mock_tool_actor("search")
        parser = OpenInstructLegacyToolParser([mock_actor])

        result = parser.format_tool_outputs(["Result 1", "Result 2"])
        expected = "<output>\nResult 1\n</output>\n\n<output>\nResult 2\n</output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_custom_wrap_name(self):
        """Test formatting with custom output wrap name."""
        mock_actor = create_mock_tool_actor("search")
        parser = OpenInstructLegacyToolParser([mock_actor], output_wrap_name="result")

        result = parser.format_tool_outputs(["Some output"])
        expected = "<result>\nSome output\n</result>\n"
        self.assertEqual(result, expected)

    def test_stop_sequences(self):
        """Test that stop sequences are correctly generated."""
        mock_search = create_mock_tool_actor("search")
        mock_code = create_mock_tool_actor("code")
        parser = OpenInstructLegacyToolParser([mock_search, mock_code])

        stop_seqs = parser.stop_sequences()

        self.assertEqual(len(stop_seqs), 2)
        self.assertIn("</search>", stop_seqs)
        self.assertIn("</code>", stop_seqs)

    def test_empty_content(self):
        """Test tool call with empty content between tags."""
        mock_actor = create_mock_tool_actor("search", param_name="query")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "Empty search: <search></search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "")

    def test_whitespace_only_content(self):
        """Test tool call with whitespace-only content."""
        mock_actor = create_mock_tool_actor("search", param_name="query")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "Whitespace: <search>   \n\t  </search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "   \n\t  ")

    def test_tool_without_required_params_uses_first_property(self):
        """Test that tools without required params use first property name."""
        mock_actor = create_mock_tool_actor("search", param_name="query", required=[])
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "<search>test query</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "test query")

    def test_multiple_calls_same_tool_extracted(self):
        """Test that all occurrences of the same tool type are extracted."""
        mock_actor = create_mock_tool_actor("search", param_name="query")
        parser = OpenInstructLegacyToolParser([mock_actor])

        text = "<search>first query</search> then <search>second query</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].args["query"], "first query")
        self.assertEqual(tool_calls[1].args["query"], "second query")

    def test_tool_calls_preserve_text_order(self):
        """Test that tool calls are returned in the order they appear in text."""
        mock_search = create_mock_tool_actor("search", param_name="query")
        mock_code = create_mock_tool_actor("code", param_name="script")
        parser = OpenInstructLegacyToolParser([mock_search, mock_code])

        # Interleaved tool calls: code, search, code
        text = "<code>first code</code> then <search>query</search> then <code>second code</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 3)
        self.assertEqual(tool_calls[0].name, "code")
        self.assertEqual(tool_calls[0].args["script"], "first code")
        self.assertEqual(tool_calls[1].name, "search")
        self.assertEqual(tool_calls[1].args["query"], "query")
        self.assertEqual(tool_calls[2].name, "code")
        self.assertEqual(tool_calls[2].args["script"], "second code")

    def test_special_regex_characters_in_tool_name(self):
        """Test that tool names with regex special chars are properly escaped."""
        # Tool name with characters that have meaning in regex
        mock_actor = create_mock_tool_actor("tool.name", param_name="input")
        parser = OpenInstructLegacyToolParser([mock_actor])

        # Should match literal <tool.name> not <toolXname>
        text = "<tool.name>content</tool.name>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["input"], "content")


class TestGetAvailableParsers(unittest.TestCase):
    """Tests for get_available_parsers function."""

    def test_returns_list(self):
        """Test that available parsers returns a list."""
        parsers = get_available_parsers()
        self.assertIsInstance(parsers, list)

    def test_contains_legacy(self):
        """Test that legacy parser is available."""
        parsers = get_available_parsers()
        self.assertIn("legacy", parsers)


if __name__ == "__main__":
    unittest.main()
