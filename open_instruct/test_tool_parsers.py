import json
import re
import unittest
from unittest import mock

import transformers
from vllm.entrypoints.openai import protocol as vllm_protocol
from vllm.entrypoints.openai.tool_parsers import hermes_tool_parser
from vllm.entrypoints.openai.tool_parsers import pythonic_tool_parser

from open_instruct.queue_types import SamplingConfig
from open_instruct.tool_utils.tools import MaxCallsExceededTool, PythonCodeTool, Tool, ToolOutput


def get_triggered_tool(
    output_text: str,
    tools: dict[str, Tool],
    max_tool_calls: dict[str, int],
    num_calls: int,
    sampling_params: SamplingConfig,
) -> tuple[Tool | None, str | None]:
    """Original get_triggered_tool from main branch."""
    if not sampling_params.stop:
        return None, None

    for stop_str in sampling_params.stop:
        if stop_str in tools and output_text.endswith(stop_str):
            if num_calls < max_tool_calls.get(stop_str, 0):
                return tools[stop_str], stop_str
            else:
                return MaxCallsExceededTool(start_str="<tool>", end_str="</tool>"), stop_str
    return None, None


class MockPythonCodeTool(PythonCodeTool):
    def __init__(self):
        self.start_str = "<code>"
        self.end_str = "</code>"
        self.name = "code"

    def __call__(self, input_text: str) -> ToolOutput:
        re_str = r"(?s)(?<!`)<code>\s*(.*?)\s*</code>"
        code_blocks = re.findall(re_str, input_text, re.DOTALL)
        code = code_blocks[-1] if code_blocks else input_text.strip()
        return ToolOutput(output=f"Extracted code: {code}", called=True, error="", timeout=False, runtime=0.1)


class TestToolParserFormats(unittest.TestCase):
    """Demonstrate the formats supported by each tool parser."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    def test_hermes_format(self):
        """Hermes2Pro uses XML tags with JSON content."""
        output = """I'll help you with that calculation.
<tool_call>
{"name": "code", "arguments": {"input": "print(2 + 2)"}}
</tool_call>"""

        parser = hermes_tool_parser.Hermes2ProToolParser(self.tokenizer)
        request = mock.MagicMock(spec=vllm_protocol.ChatCompletionRequest)
        result = parser.extract_tool_calls(output, request)

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "code")
        self.assertEqual(json.loads(result.tool_calls[0].function.arguments), {"input": "print(2 + 2)"})

    def test_olmo3_format(self):
        """OLMo3 uses Python-like function call syntax."""
        output = '[code(input="print(2 + 2)")]'

        parser = pythonic_tool_parser.PythonicToolParser(self.tokenizer)
        request = mock.MagicMock(spec=vllm_protocol.ChatCompletionRequest)
        result = parser.extract_tool_calls(output, request)

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "code")
        self.assertEqual(json.loads(result.tool_calls[0].function.arguments), {"input": "print(2 + 2)"})

    def test_get_triggered_tool_format(self):
        """get_triggered_tool uses stop-string suffix matching with <code> tags."""
        output = """Here's the code:
<code>
print(2 + 2)
</code>"""

        tool = MockPythonCodeTool()
        tools = {"</code>": tool}
        max_tool_calls = {"</code>": 5}
        sampling_params = SamplingConfig(stop=["</code>"])

        matched_tool, stop_str = get_triggered_tool(output, tools, max_tool_calls, 0, sampling_params)
        self.assertIsNotNone(matched_tool)
        self.assertEqual(stop_str, "</code>")

        result = matched_tool(output)
        self.assertEqual(result.output, "Extracted code: print(2 + 2)")


if __name__ == "__main__":
    unittest.main()
