"""Single-step reference-action rewards; tool calls are parsed but never executed."""

import copy
import json
import re
from functools import lru_cache

from jsonschema import validators
from jsonschema.exceptions import SchemaError
from pydantic import TypeAdapter
from referencing import Registry
from referencing.exceptions import Unresolvable

from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import nemo_tool_comparison as native
from open_instruct.miles.rewards import nvidia_constraints, response_text, tool_call_parsing

UPSTREAM_REVISION = "966145fc02ae58b887fd49379675604cc38fdd43"
ROUTES = {
    "single_step_tool_use_with_argument_comparison_agent": "conversational",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": "function_calling",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent": "swe_pivot",
}
THRESHOLDS = {"conversational": 0.1, "function_calling": 0.1, "swe_pivot": 0.0}


def json_object(value):
    if isinstance(value, str):
        value = json.loads(
            value, object_pairs_hook=tool_call_parsing._unique_keys, parse_constant=tool_call_parsing._invalid_constant
        )
    if not isinstance(value, dict):
        raise ValueError("Function arguments must be a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def declared_tools(raw):
    tools = (raw.get("responses_create_params") or {}).get("tools")
    if not isinstance(tools, list):
        raise ValueError("Single-step tool tasks require an explicit tools array")
    result, names = [], set()
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ValueError("Only declared function tools are supported")
        function = copy.deepcopy(item.get("function", {k: v for k, v in item.items() if k != "type"}))
        name = function.get("name")
        if not isinstance(name, str) or not name.strip() or re.search(r"[<>\n]", name) or name in names:
            raise ValueError("Function names must be distinct text without markup delimiters")
        names.add(name)
        function["parameters"] = json_object(function.get("parameters"))
        result.append({"type": "function", "function": function})
    return result


@lru_cache(maxsize=4096)
def contract(serialized):
    target = json.loads(serialized)
    if set(target) != {"profile", "tool_format", "expected_action", "tools"}:
        raise ValueError("Unexpected single-step tool target fields")
    if target["profile"] not in THRESHOLDS:
        raise ValueError("Unknown native tool comparison profile")
    if target["tool_format"] not in {"olmo_pythonic_v1", "olmo_xml_v1"}:
        raise ValueError("Unknown native tool rendering format")
    expected = TypeAdapter(native.ExpectedAction).validate_python(target["expected_action"])
    names = {t["function"]["name"] for t in target["tools"]}
    for call in native.get_tool_calls(expected):
        if call.name not in names:
            raise ValueError("Expected call names an undeclared tool")
        json_object(call.arguments)
    config = native.ToolCallComparatorConfig(word_count_similarity_threshold=THRESHOLDS[target["profile"]])
    return target, expected, config


def target_for_row(raw, options):
    agent = raw.get("agent_ref")
    route = agent.get("name") if isinstance(agent, dict) else agent
    profile = options.get("profile", ROUTES.get(route))
    target = {
        "profile": profile,
        "tool_format": options.get("tool_format", "olmo_pythonic_v1"),
        "expected_action": copy.deepcopy(raw["expected_action"]),
        "tools": declared_tools(raw),
    }
    contract(json.dumps(target, allow_nan=False))
    if target["tool_format"] == "olmo_xml_v1":
        validate_xml_reference(target)
    return target


def action_text(prediction, prompt_opens_reasoning=False):
    """Return the response's action region, or None while its reasoning is still open.

    NeMo Gym receives reasoning as a separate output item (vLLM's Nemotron V3 and
    DeepSeek-R1 reasoning parsers treat output without a closing tag as all reasoning)
    and grades only the assistant message or calls that follow it. Reasoning that never
    closes therefore contains no action, and prose or calls written inside it never
    count. When the prompt ends with the opening tag, the response does not repeat it.
    """
    if "</think>" in prediction:
        final = prediction.rsplit("</think>", 1)[1]
    elif prompt_opens_reasoning or "<think>" in prediction:
        return None
    else:
        final = prediction
    # Conservatively treat any literal opener in the final region, including
    # quoted tags or tool-argument text, as reopened and unfinished reasoning.
    if "<think>" in final:
        return None
    return common.answer_text(final)


def parse_action(text, target):
    text = response_text.strip_terminal_special_tokens(text).strip()
    calls = []
    if target["tool_format"] == "olmo_pythonic_v1" and "<function_calls>" in text:
        blocks = re.findall(r"<function_calls>(.*?)</function_calls>", text, re.S)
        if len(blocks) != text.count("<function_calls>") or len(blocks) != text.count("</function_calls>"):
            raise ValueError("Unclosed function-call block")
        for block in blocks:
            for call in tool_call_parsing.parse_pythonic_calls(
                "<function_calls>" + block + "</function_calls>", target["tools"]
            ):
                calls.append(
                    native.FunctionCallAction(
                        type="function_call",
                        name=call["name"],
                        arguments=json.dumps(call["parameters"], allow_nan=False),
                    )
                )
    elif target["tool_format"] == "olmo_xml_v1" and "<tool_call>" in text:
        blocks = re.findall(r"<tool_call>.*?</tool_call>", text, re.S)
        if len(blocks) != text.count("<tool_call>") or len(blocks) != text.count("</tool_call>"):
            raise ValueError("Unclosed tool-call block")
        for call in tool_call_parsing.parse_native_calls("\n".join(blocks), target["tools"]):
            calls.append(
                native.FunctionCallAction(
                    type="function_call", name=call["name"], arguments=json.dumps(call["parameters"], allow_nan=False)
                )
            )
    elif re.search(r"</?(?:function_calls|tool_call|function|parameter)\b", text):
        raise ValueError("Malformed or incompatible tool-call format")
    if len(calls) == 1:
        return calls[0]
    if calls:
        return native.FunctionCallBatchAction(type="function_call_batch", calls=calls)
    if re.search(r"</?(?:function_calls|tool_call)>", text):
        raise ValueError("Empty function-call block")
    return native.MessageAction(type="message", content=text) if text else None


@lru_cache(maxsize=256)
def _reference_validator(serialized_schema):
    schema = json.loads(serialized_schema)
    cls = validators.validator_for(schema)
    try:
        cls.check_schema(schema)
    except SchemaError as error:
        raise ValueError("Invalid declared tool schema") from error
    return cls(schema, registry=Registry())


def validate_xml_reference(target):
    """Admit only references representable by the declared schemas and XML grammar."""
    _, expected, config = contract(json.dumps(target, allow_nan=False))
    tools = {tool["function"]["name"]: tool["function"] for tool in target["tools"]}
    # Even tools unused by the reference may be called by a model response.
    for tool in tools.values():
        _reference_validator(json.dumps(tool["parameters"], sort_keys=True, allow_nan=False))
    blocks = []
    for call in native.get_tool_calls(expected):
        arguments = json_object(call.arguments)
        schema = tools[call.name]["parameters"]
        validator = _reference_validator(json.dumps(schema, sort_keys=True, allow_nan=False))
        try:
            if not validator.is_valid(arguments):
                raise ValueError("Reference arguments violate the declared tool schema")
        except Unresolvable as error:
            raise ValueError("Reference schema contains an unavailable resource") from error
        parameters = "".join(
            f"<parameter={name}>\n{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)}\n</parameter>\n"
            for name, value in arguments.items()
        )
        blocks.append(f"<tool_call>\n<function={call.name}>\n{parameters}</function>\n</tool_call>")
    if blocks:
        actual = parse_action("\n".join(blocks), target)
        if native.ActionComparator(config=config).compare_action(expected, actual).reward != 1:
            raise ValueError("Reference call does not round-trip through the XML parser")


class SingleStepToolVerifier(nvidia_constraints._IsolatedAdapter):
    kind = "nemo_single_step_tool_v1"
    # The reward step tells this verifier whether the rendered prompt opened reasoning.
    reads_prompt_reasoning = True

    def __call__(self, tokenized_prediction, prediction, label, prompt_opens_reasoning=False, **kwargs):
        """Grade an action; direct callers must supply the prompt's reasoning state.

        Set prompt_opens_reasoning=True when the rendered prompt ends in <think>.
        Without that state, tag-less text is treated as a non-thinking response.
        registered_reward supplies the state automatically for string prompts.
        """
        target, expected, config = contract(json.dumps(label, allow_nan=False))
        text = action_text(prediction, prompt_opens_reasoning)
        if text is None:
            return self.result(0.0, category="unclosed_thinking")
        try:
            actual = parse_action(text, target)
            if actual is None:
                return self.result(0.0, category="no_action")
            result = native.ActionComparator(config=config).compare_action(expected, actual)
            return self.result(result.reward, category=result.category.value)
        except (ValueError, SyntaxError, TypeError, RecursionError, OverflowError) as error:
            return self.result(0.0, status="invalid_response", error=str(error))
