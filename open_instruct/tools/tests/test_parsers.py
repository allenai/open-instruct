"""Tests for tool parsers."""

import unittest
from unittest.mock import MagicMock, patch

from parameterized import parameterized

from open_instruct.tools.parsers import (
    VLLM_PARSERS,
    DRTuluToolParser,
    OpenInstructLegacyToolParser,
    VllmParserConfig,
    VllmToolParser,
    create_tool_parser,
    get_available_parsers,
)
from open_instruct.utils import import_class_from_string


class MockTool:
    """Mock tool for testing without ray."""

    def __init__(
        self,
        name: str,
        param_name: str = "text",
        required: list[str] | None = None,
        stop_strings: list[str] | None = None,
    ):
        self.call_name = name
        self.param_name = param_name
        self.required = required if required is not None else [param_name]
        self._stop_strings = stop_strings

    def get_call_name(self):
        return self.call_name

    def get_parameters(self):
        return {"required": self.required, "properties": {self.param_name: {"type": "string"}}}

    def get_stop_strings(self):
        if self._stop_strings is not None:
            return self._stop_strings
        raise AttributeError("No stop_strings defined")


def create_mock_tool_actor(
    name: str, param_name: str = "text", required: list[str] | None = None, stop_strings: list[str] | None = None
) -> MagicMock:
    """Create a mock tool actor handle that works with ray.get()."""
    mock_tool = MockTool(name, param_name, required, stop_strings)
    actor_handle = MagicMock()
    actor_handle.get_call_name.remote.return_value = mock_tool.get_call_name()
    actor_handle.get_parameters.remote.return_value = mock_tool.get_parameters()

    # Handle get_stop_strings - either return value or raise AttributeError
    if stop_strings is not None:
        actor_handle.get_stop_strings.remote.return_value = mock_tool.get_stop_strings()
    else:
        actor_handle.get_stop_strings.remote.side_effect = AttributeError("No stop_strings defined")

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

        stop_seqs = parser.stop_sequences

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


class TestDRTuluToolParser(unittest.TestCase):
    """Tests for DRTuluToolParser."""

    def setUp(self):
        """Set up mock actors for each test."""
        self.patcher = patch("open_instruct.tools.parsers.ray")
        self.mock_ray = self.patcher.start()
        # Make ray.get return the value directly (simulating sync behavior)
        self.mock_ray.get.side_effect = lambda x: x
        # Mock exceptions module
        self.mock_ray.exceptions.RayActorError = Exception

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_single_tool_extraction(self):
        """Test extracting a single tool call with name attribute."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = 'I need to search. <call_tool name="google_search">python tutorials</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "google_search")
        self.assertEqual(tool_calls[0].args, {"query": "python tutorials"})

    def test_single_quotes_in_name_attribute(self):
        """Test extracting tool call with single quotes around name."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = "Search: <call_tool name='google_search'>climate change</call_tool>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "google_search")
        self.assertEqual(tool_calls[0].args, {"query": "climate change"})

    def test_multiple_tools_extraction(self):
        """Test extracting multiple different tool calls."""
        mock_search = create_mock_tool_actor("google_search", param_name="query")
        mock_browse = create_mock_tool_actor("browse_webpage", param_name="url")
        parser = DRTuluToolParser([mock_search, mock_browse])

        text = """<call_tool name="google_search">AI research</call_tool>
        Found a link. <call_tool name="browse_webpage">https://example.com</call_tool>"""
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "google_search")
        self.assertEqual(tool_calls[0].args["query"], "AI research")
        self.assertEqual(tool_calls[1].name, "browse_webpage")
        self.assertEqual(tool_calls[1].args["url"], "https://example.com")

    def test_no_tool_calls(self):
        """Test that no tool calls are returned when none exist."""
        mock_actor = create_mock_tool_actor("google_search")
        parser = DRTuluToolParser([mock_actor])

        text = "This is just regular text without any tool calls."
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_multiline_content(self):
        """Test extracting tool calls with multiline content."""
        mock_actor = create_mock_tool_actor("snippet_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        query_content = """machine learning
retrieval augmented generation
2024 papers"""
        text = f'<call_tool name="snippet_search">{query_content}</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "snippet_search")
        self.assertEqual(tool_calls[0].args["query"], query_content)

    def test_partial_tag_not_matched(self):
        """Test that incomplete tags are not matched."""
        mock_actor = create_mock_tool_actor("google_search")
        parser = DRTuluToolParser([mock_actor])

        # Missing closing tag
        text = '<call_tool name="google_search">query without closing'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_unknown_tool_name_skipped(self):
        """Test that unknown tool names are skipped with a warning."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = '<call_tool name="unknown_tool">some query</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
        mock_actor = create_mock_tool_actor("google_search")
        parser = DRTuluToolParser([mock_actor])

        result = parser.format_tool_outputs(["Search result: Found 5 items"])
        expected = "<tool_output>\nSearch result: Found 5 items\n</tool_output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_multiple(self):
        """Test formatting multiple tool outputs."""
        mock_actor = create_mock_tool_actor("google_search")
        parser = DRTuluToolParser([mock_actor])

        result = parser.format_tool_outputs(["Result 1", "Result 2"])
        expected = "<tool_output>\nResult 1\n</tool_output>\n\n<tool_output>\nResult 2\n</tool_output>\n"
        self.assertEqual(result, expected)

    def test_stop_sequences_default(self):
        """Test that default stop sequence is used when tools don't provide them."""
        mock_actor = create_mock_tool_actor("google_search")
        parser = DRTuluToolParser([mock_actor])

        self.assertEqual(parser.stop_sequences, ["</call_tool>"])

    def test_stop_sequences_from_tools(self):
        """Test that stop sequences are collected from tools that provide them."""
        mock_actor = create_mock_tool_actor("mcp", param_name="text", stop_strings=["</call_tool>", "</tool>"])
        parser = DRTuluToolParser([mock_actor])

        self.assertEqual(parser.stop_sequences, ["</call_tool>", "</tool>"])

    def test_stop_sequences_deduplicated(self):
        """Test that duplicate stop sequences are removed."""
        mock_actor1 = create_mock_tool_actor("tool1", stop_strings=["</call_tool>", "</tool>"])
        mock_actor2 = create_mock_tool_actor("tool2", stop_strings=["</call_tool>", "</other>"])
        parser = DRTuluToolParser([mock_actor1, mock_actor2])

        # Should be deduplicated while preserving order
        self.assertEqual(parser.stop_sequences, ["</call_tool>", "</tool>", "</other>"])

    def test_empty_content(self):
        """Test tool call with empty content between tags."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = '<call_tool name="google_search"></call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "")

    def test_whitespace_only_content(self):
        """Test tool call with whitespace-only content."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = '<call_tool name="google_search">   \n\t  </call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "   \n\t  ")

    def test_tool_without_required_params_uses_first_property(self):
        """Test that tools without required params use first property name."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query", required=[])
        parser = DRTuluToolParser([mock_actor])

        text = '<call_tool name="google_search">test query</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "test query")

    def test_multiple_calls_same_tool_extracted(self):
        """Test that all occurrences of the same tool type are extracted."""
        mock_actor = create_mock_tool_actor("google_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        text = """<call_tool name="google_search">first query</call_tool>
        <call_tool name="google_search">second query</call_tool>"""
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].args["query"], "first query")
        self.assertEqual(tool_calls[1].args["query"], "second query")

    def test_tool_calls_preserve_text_order(self):
        """Test that tool calls are returned in the order they appear in text."""
        mock_search = create_mock_tool_actor("google_search", param_name="query")
        mock_browse = create_mock_tool_actor("browse_webpage", param_name="url")
        mock_snippet = create_mock_tool_actor("snippet_search", param_name="query")
        parser = DRTuluToolParser([mock_search, mock_browse, mock_snippet])

        # Interleaved tool calls
        text = """<call_tool name="google_search">first</call_tool>
        <call_tool name="browse_webpage">https://example.com</call_tool>
        <call_tool name="google_search">second</call_tool>"""
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 3)
        self.assertEqual(tool_calls[0].name, "google_search")
        self.assertEqual(tool_calls[0].args["query"], "first")
        self.assertEqual(tool_calls[1].name, "browse_webpage")
        self.assertEqual(tool_calls[1].args["url"], "https://example.com")
        self.assertEqual(tool_calls[2].name, "google_search")
        self.assertEqual(tool_calls[2].args["query"], "second")

    def test_extra_attributes_in_tag(self):
        """Test that extra attributes in the call_tool tag are handled."""
        mock_actor = create_mock_tool_actor("snippet_search", param_name="query")
        parser = DRTuluToolParser([mock_actor])

        # DR Tulu format supports extra attributes like limit, year, etc.
        text = '<call_tool name="snippet_search" limit="5" year="2024">machine learning</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "snippet_search")
        self.assertEqual(tool_calls[0].args["query"], "machine learning")

    def test_realistic_dr_tulu_format(self):
        """Test with realistic DR Tulu format including think tags."""
        mock_search = create_mock_tool_actor("google_search", param_name="query")
        mock_snippet = create_mock_tool_actor("snippet_search", param_name="query")
        parser = DRTuluToolParser([mock_search, mock_snippet])

        text = """<think>I need to find information about climate change effects.</think>
<call_tool name="google_search">climate change effects 2024</call_tool>

<think>Now I need to find scientific papers on this topic.</think>
<call_tool name="snippet_search" limit="5" year="2023-2024">climate change impact on agriculture</call_tool>"""

        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "google_search")
        self.assertEqual(tool_calls[0].args["query"], "climate change effects 2024")
        self.assertEqual(tool_calls[1].name, "snippet_search")
        self.assertEqual(tool_calls[1].args["query"], "climate change impact on agriculture")


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

    def test_contains_dr_tulu(self):
        """Test that dr_tulu parser is available."""
        parsers = get_available_parsers()
        self.assertIn("dr_tulu", parsers)

    def test_contains_vllm_parsers(self):
        """Test that vLLM parsers are in the list."""
        parsers = get_available_parsers()
        self.assertIn("vllm_hermes", parsers)
        self.assertIn("vllm_llama3_json", parsers)
        self.assertIn("vllm_olmo3", parsers)


class TestVllmParserRegistry(unittest.TestCase):
    """Tests for vLLM parser registry and helpers."""

    @parameterized.expand(VLLM_PARSERS.items())
    def test_vllm_parser_config(self, name, config):
        """Test that a registered vLLM parser has valid configuration."""
        # Check config type
        self.assertIsInstance(config, VllmParserConfig)

        # Verify import_path resolves to a callable class
        self.assertTrue(config.import_path, "missing import_path")
        parser_cls = import_class_from_string(config.import_path)
        self.assertTrue(callable(parser_cls))

        # Verify output_template is usable with .format()
        self.assertTrue(config.output_template, "missing output_template")
        formatted = config.output_template.format("test_output")
        self.assertIn("test_output", formatted)

        # Check stop_sequences is a sized iterable (list, tuple, set, etc.)
        self.assertGreaterEqual(len(config.stop_sequences), 0, "stop_sequences must be a sized iterable")


class TestVllmToolParser(unittest.TestCase):
    """Tests for VllmToolParser class."""

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
        # Create a mock native parser (we don't need it for format tests)
        mock_native = MagicMock()
        parser = VllmToolParser(
            tool_parser=mock_native,
            output_formatter=lambda x: f"<result>{x}</result>",
            stop_sequences=["</tool_call>"],
            output_prefix="",
            output_postfix="<|assistant|>",
        )

        result = parser.format_tool_outputs(["test output"])
        self.assertEqual(result, "<result>test output</result><|assistant|>")

    def test_format_tool_outputs_multiple(self):
        """Test formatting multiple tool outputs."""
        mock_native = MagicMock()
        parser = VllmToolParser(
            tool_parser=mock_native,
            output_formatter=lambda x: f"<result>{x}</result>\n",
            stop_sequences=[],
            output_prefix="<|tools|>",
            output_postfix="<|assistant|>",
        )

        result = parser.format_tool_outputs(["output1", "output2"])
        self.assertEqual(result, "<|tools|><result>output1</result>\n<result>output2</result>\n<|assistant|>")

    def test_stop_sequences_empty_by_default(self):
        """Test that stop sequences can be empty for vLLM parsers."""
        mock_native = MagicMock()
        parser = VllmToolParser(tool_parser=mock_native, output_formatter=lambda x: x, stop_sequences=[])
        self.assertEqual(parser.stop_sequences, [])

    def test_stop_sequences_custom(self):
        """Test custom stop sequences."""
        mock_native = MagicMock()
        parser = VllmToolParser(
            tool_parser=mock_native, output_formatter=lambda x: x, stop_sequences=["</tool>", "<|end|>"]
        )
        self.assertEqual(parser.stop_sequences, ["</tool>", "<|end|>"])


class TestCreateToolParser(unittest.TestCase):
    """Tests for create_tool_parser factory function."""

    def setUp(self):
        """Set up mock actors for each test."""
        self.patcher = patch("open_instruct.tools.parsers.ray")
        self.mock_ray = self.patcher.start()
        self.mock_ray.get.side_effect = lambda x: x
        self.mock_ray.exceptions.RayActorError = Exception

    def tearDown(self):
        """Stop the patcher."""
        self.patcher.stop()

    def test_create_legacy_parser(self):
        """Test creating legacy parser."""
        mock_actor = create_mock_tool_actor("search")
        mock_tokenizer = MagicMock()

        parser = create_tool_parser("legacy", tokenizer=mock_tokenizer, tool_actors=[mock_actor])
        self.assertIsInstance(parser, OpenInstructLegacyToolParser)

    def test_create_dr_tulu_parser(self):
        """Test creating dr_tulu parser."""
        mock_actor = create_mock_tool_actor("search")
        mock_tokenizer = MagicMock()

        parser = create_tool_parser("dr_tulu", tokenizer=mock_tokenizer, tool_actors=[mock_actor])
        self.assertIsInstance(parser, DRTuluToolParser)

    @parameterized.expand([(p,) for p in VLLM_PARSERS])
    def test_create_vllm_parser(self, parser_type):
        """Test creating vLLM parsers."""
        mock_actor = create_mock_tool_actor("search")
        mock_tokenizer = MagicMock()

        # Mock the vLLM parser class import
        with patch("open_instruct.tools.parsers.import_class_from_string") as mock_import:
            mock_import.return_value = MagicMock()
            parser = create_tool_parser(parser_type, tokenizer=mock_tokenizer, tool_actors=[mock_actor])
            self.assertIsInstance(parser, VllmToolParser)

    def test_unknown_parser_raises_error(self):
        """Test that unknown parser types raise an error."""
        mock_actor = create_mock_tool_actor("search")
        mock_tokenizer = MagicMock()
        with self.assertRaises(ValueError) as context:
            create_tool_parser("unknown_parser", tokenizer=mock_tokenizer, tool_actors=[mock_actor])
        self.assertIn("Unknown parser type", str(context.exception))
        self.assertIn("Available:", str(context.exception))


if __name__ == "__main__":
    unittest.main()
