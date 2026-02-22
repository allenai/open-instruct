"""Tests for tool parsers."""

import unittest
from unittest.mock import MagicMock, patch

from parameterized import parameterized

from open_instruct.environments.tools.parsers import (
    VLLM_PARSERS,
    DRTuluToolParser,
    OpenInstructLegacyToolParser,
    VllmParserConfig,
    VllmToolParser,
    create_tool_parser,
    get_available_parsers,
)
from open_instruct.utils import import_class_from_string


def make_tool_definition(name: str, param_name: str = "text", required: list[str] | None = None) -> dict:
    """Create an OpenAI-format tool definition for testing."""
    if required is None:
        required = [param_name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {
                "type": "object",
                "properties": {param_name: {"type": "string", "description": f"The {param_name} parameter"}},
                "required": required,
            },
        },
    }


class TestOpenInstructLegacyToolParser(unittest.TestCase):
    """Tests for OpenInstructLegacyToolParser."""

    def test_single_tool_extraction(self):
        """Test extracting a single tool call."""
        defs = [make_tool_definition("search", param_name="query")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "I need to search for something. <search>python tutorials</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "search")
        self.assertEqual(tool_calls[0].args, {"query": "python tutorials"})

    def test_multiple_tools_extraction(self):
        """Test extracting multiple different tool calls."""
        defs = [make_tool_definition("search", param_name="query"), make_tool_definition("code", param_name="script")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "First search <search>python</search> then run <code>print('hello')</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        tool_names = {tc.name for tc in tool_calls}
        self.assertEqual(tool_names, {"search", "code"})

    def test_no_tool_calls(self):
        """Test that no tool calls are returned when none exist."""
        defs = [make_tool_definition("search")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "This is just regular text without any tool calls."
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_multiline_content(self):
        """Test extracting tool calls with multiline content."""
        defs = [make_tool_definition("code", param_name="script")]
        parser = OpenInstructLegacyToolParser(defs)

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
        defs = [make_tool_definition("search")]
        parser = OpenInstructLegacyToolParser(defs)

        # Missing closing tag
        text = "Here's a search <search>query without closing"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_nested_content_with_angle_brackets(self):
        """Test content containing angle brackets."""
        defs = [make_tool_definition("code", param_name="script")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "<code>if x > 5 and y < 10: print('yes')</code>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["script"], "if x > 5 and y < 10: print('yes')")

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
        defs = [make_tool_definition("search")]
        parser = OpenInstructLegacyToolParser(defs)

        result = parser.format_tool_outputs(["Search result: Found 5 items"])
        expected = "<output>\nSearch result: Found 5 items\n</output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_multiple(self):
        """Test formatting multiple tool outputs."""
        defs = [make_tool_definition("search")]
        parser = OpenInstructLegacyToolParser(defs)

        result = parser.format_tool_outputs(["Result 1", "Result 2"])
        expected = "<output>\nResult 1\n</output>\n\n<output>\nResult 2\n</output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_custom_wrap_name(self):
        """Test formatting with custom output wrap name."""
        defs = [make_tool_definition("search")]
        parser = OpenInstructLegacyToolParser(defs, output_wrap_name="result")

        result = parser.format_tool_outputs(["Some output"])
        expected = "<result>\nSome output\n</result>\n"
        self.assertEqual(result, expected)

    def test_stop_sequences(self):
        """Test that stop sequences are correctly generated."""
        defs = [make_tool_definition("search"), make_tool_definition("code")]
        parser = OpenInstructLegacyToolParser(defs)

        stop_seqs = parser.stop_sequences

        self.assertEqual(len(stop_seqs), 2)
        self.assertIn("</search>", stop_seqs)
        self.assertIn("</code>", stop_seqs)

    def test_empty_content(self):
        """Test tool call with empty content between tags."""
        defs = [make_tool_definition("search", param_name="query")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "Empty search: <search></search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "")

    def test_whitespace_only_content(self):
        """Test tool call with whitespace-only content."""
        defs = [make_tool_definition("search", param_name="query")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "Whitespace: <search>   \n\t  </search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "   \n\t  ")

    def test_tool_without_required_params_uses_first_property(self):
        """Test that tools without required params use first property name."""
        defs = [make_tool_definition("search", param_name="query", required=[])]
        parser = OpenInstructLegacyToolParser(defs)

        text = "<search>test query</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["query"], "test query")

    def test_multiple_calls_same_tool_extracted(self):
        """Test that all occurrences of the same tool type are extracted."""
        defs = [make_tool_definition("search", param_name="query")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "<search>first query</search> then <search>second query</search>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].args["query"], "first query")
        self.assertEqual(tool_calls[1].args["query"], "second query")

    def test_tool_calls_preserve_text_order(self):
        """Test that tool calls are returned in the order they appear in text."""
        defs = [make_tool_definition("search", param_name="query"), make_tool_definition("code", param_name="script")]
        parser = OpenInstructLegacyToolParser(defs)

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
        defs = [make_tool_definition("tool.name", param_name="input")]
        parser = OpenInstructLegacyToolParser(defs)

        text = "<tool.name>content</tool.name>"
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["input"], "content")

    def test_no_definitions(self):
        """Test parser with no tool definitions."""
        parser = OpenInstructLegacyToolParser()

        self.assertEqual(parser.tool_names, [])
        self.assertEqual(parser.stop_sequences, [])
        self.assertEqual(parser.get_tool_calls("any text"), [])


class TestDRTuluToolParser(unittest.TestCase):
    """Tests for DRTuluToolParser.

    The DRTuluToolParser delegates actual parsing to the tool itself.
    It only detects that a tool call occurred (via stop strings) and passes the full text.
    """

    DR_AGENT_DEF = make_tool_definition("dr_agent_mcp")

    def test_detects_tool_call_with_stop_string(self):
        """Test that parser detects tool call when stop string is present."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        text = '<call_tool name="google_search">python tutorials</call_tool>'
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "dr_agent_mcp")
        self.assertEqual(tool_calls[0].args, {"text": text})

    def test_no_tool_call_without_stop_string(self):
        """Test that no tool call is returned when stop string is absent."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        text = "This is just regular text without any tool calls."
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 0)

    def test_passes_full_text_to_tool(self):
        """Test that the full text is passed as the argument."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        text = """<think>I need to search</think>
<call_tool name="google_search">query here</call_tool>"""
        tool_calls = parser.get_tool_calls(text)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].args["text"], text)

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        result = parser.format_tool_outputs(["Search result: Found 5 items"])
        expected = "<tool_output>\nSearch result: Found 5 items\n</tool_output>\n"
        self.assertEqual(result, expected)

    def test_format_tool_outputs_multiple(self):
        """Test formatting multiple tool outputs."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        result = parser.format_tool_outputs(["Result 1", "Result 2"])
        expected = "<tool_output>\nResult 1\n</tool_output>\n\n<tool_output>\nResult 2\n</tool_output>\n"
        self.assertEqual(result, expected)

    def test_stop_sequences_empty(self):
        """Test that empty list is used when no stop sequences provided."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=[])

        self.assertEqual(parser.stop_sequences, [])

    def test_stop_sequences_from_init(self):
        """Test that stop sequences are set from init parameter."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>", "</tool>"])

        self.assertEqual(parser.stop_sequences, ["</call_tool>", "</tool>"])

    def test_stop_sequences_deduplicated(self):
        """Test that duplicate stop sequences are removed."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>", "</tool>", "</call_tool>"])

        self.assertEqual(parser.stop_sequences, ["</call_tool>", "</tool>"])

    def test_rejects_multiple_tools(self):
        """Test that parser rejects multiple tools."""
        defs = [self.DR_AGENT_DEF, make_tool_definition("other_tool")]

        with self.assertRaises(ValueError) as context:
            DRTuluToolParser(defs, stop_sequences=["</call_tool>"])
        self.assertIn("exactly one tool", str(context.exception))

    def test_uses_tool_call_name(self):
        """Test that parser uses the tool's call name for routing."""
        parser = DRTuluToolParser([self.DR_AGENT_DEF], stop_sequences=["</call_tool>"])

        self.assertEqual(parser.tool_call_name, "dr_agent_mcp")

        text = '<call_tool name="google_search">query</call_tool>'
        tool_calls = parser.get_tool_calls(text)
        self.assertEqual(tool_calls[0].name, "dr_agent_mcp")

    def test_rejects_wrong_tool(self):
        """Test that parser rejects tools that aren't dr_agent_mcp."""
        defs = [make_tool_definition("python")]

        with self.assertRaises(ValueError) as context:
            DRTuluToolParser(defs, stop_sequences=["</code>"])
        self.assertIn("dr_agent_mcp", str(context.exception))


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
        self.assertIsInstance(config, VllmParserConfig)

        self.assertTrue(config.import_path, "missing import_path")
        parser_cls = import_class_from_string(config.import_path)
        self.assertTrue(callable(parser_cls))

        self.assertTrue(config.output_template, "missing output_template")
        formatted = config.output_template.format("test_output")
        self.assertIn("test_output", formatted)

        self.assertGreaterEqual(len(config.stop_sequences), 0, "stop_sequences must be a sized iterable")


class TestVllmToolParser(unittest.TestCase):
    """Tests for VllmToolParser class."""

    def test_format_tool_outputs_single(self):
        """Test formatting a single tool output."""
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

    def test_create_legacy_parser(self):
        """Test creating legacy parser."""
        mock_tokenizer = MagicMock()
        defs = [make_tool_definition("search")]

        parser = create_tool_parser("legacy", tokenizer=mock_tokenizer, tool_definitions=defs)
        self.assertIsInstance(parser, OpenInstructLegacyToolParser)

    def test_create_dr_tulu_parser(self):
        """Test creating dr_tulu parser."""
        mock_tokenizer = MagicMock()
        defs = [make_tool_definition("dr_agent_mcp")]

        parser = create_tool_parser(
            "dr_tulu", tokenizer=mock_tokenizer, tool_definitions=defs, stop_sequences=["</call_tool>"]
        )
        self.assertIsInstance(parser, DRTuluToolParser)

    @parameterized.expand([(p,) for p in VLLM_PARSERS])
    def test_create_vllm_parser(self, parser_type):
        """Test creating vLLM parsers."""
        mock_tokenizer = MagicMock()
        defs = [make_tool_definition("search")]

        with patch("open_instruct.environments.tools.parsers.import_class_from_string") as mock_import:
            mock_import.return_value = MagicMock()
            parser = create_tool_parser(parser_type, tokenizer=mock_tokenizer, tool_definitions=defs)
            self.assertIsInstance(parser, VllmToolParser)

    def test_unknown_parser_raises_error(self):
        """Test that unknown parser types raise an error."""
        mock_tokenizer = MagicMock()
        with self.assertRaises(ValueError) as context:
            create_tool_parser("unknown_parser", tokenizer=mock_tokenizer)
        self.assertIn("Unknown parser type", str(context.exception))
        self.assertIn("Available:", str(context.exception))


if __name__ == "__main__":
    unittest.main()
