import ast
import json
import re
import unittest
from dataclasses import dataclass
from typing import Any

from open_instruct.queue_types import SamplingConfig
from open_instruct.tool_utils.tools import MaxCallsExceededTool, PythonCodeTool, Tool, ToolOutput


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


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


def hermes_extract_tool_calls(output_text: str) -> list[ParsedToolCall]:
    """Hermes2Pro format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>"""
    regex = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)
    tool_calls = []
    for match in regex.findall(output_text):
        try:
            data = json.loads(match)
            args = data.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ParsedToolCall(name=data.get("name", ""), arguments=args))
        except json.JSONDecodeError:
            continue
    return tool_calls


def olmo3_extract_tool_calls(output_text: str) -> list[ParsedToolCall]:
    """OLMo3 pythonic format: [function_name(arg1="value1", arg2=123)]"""
    output_text = output_text.strip()
    output_text = re.sub(r"</?function_calls?>", "", output_text)
    if not output_text.startswith("["):
        output_text = "[" + output_text
    if not output_text.endswith("]"):
        output_text = output_text + "]"
    output_text = output_text.replace("\n", ", ")
    output_text = re.sub(r",\s*,", ",", output_text)
    output_text = re.sub(r",\s*\]", "]", output_text)

    try:
        tree = ast.parse(output_text, mode="eval")
        if not isinstance(tree.body, ast.List):
            return []
        tool_calls = []
        for node in tree.body.elts:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            args = {}
            for kw in node.keywords:
                if kw.arg:
                    args[kw.arg] = _get_ast_value(kw.value)
            tool_calls.append(ParsedToolCall(name=node.func.id, arguments=args))
        return tool_calls
    except SyntaxError:
        return []


def _get_ast_value(expr: ast.expr) -> Any:
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Name):
        return {"null": None, "true": True, "false": False}.get(expr.id, expr.id)
    if isinstance(expr, ast.List):
        return [_get_ast_value(e) for e in expr.elts]
    if isinstance(expr, ast.Dict):
        return {_get_ast_value(k): _get_ast_value(v) for k, v in zip(expr.keys, expr.values)}
    return None


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

    def test_hermes_format(self):
        """Hermes2Pro uses XML tags with JSON content."""
        output = """I'll help you with that calculation.
<tool_call>
{"name": "code", "arguments": {"input": "print(2 + 2)"}}
</tool_call>"""

        calls = hermes_extract_tool_calls(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "code")
        self.assertEqual(calls[0].arguments, {"input": "print(2 + 2)"})

        tool = MockPythonCodeTool()
        result = tool(calls[0].arguments["input"])
        self.assertEqual(result.output, "Extracted code: print(2 + 2)")

    def test_olmo3_format(self):
        """OLMo3 uses Python-like function call syntax."""
        output = '[code(input="print(2 + 2)")]'

        calls = olmo3_extract_tool_calls(output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "code")
        self.assertEqual(calls[0].arguments, {"input": "print(2 + 2)"})

        tool = MockPythonCodeTool()
        result = tool(calls[0].arguments["input"])
        self.assertEqual(result.output, "Extracted code: print(2 + 2)")

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
