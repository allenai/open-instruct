# Copyright 2026 Allen Institute for Artificial Intelligence
# SPDX-License-Identifier: Apache-2.0
"""Native XML parser reused from the ToolRL integration (5732f2ef5).

Only parsing is shared; source-specific reward semantics belong to each verifier.
"""

import ast
import json
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from jsonschema import validators
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _literal(node):
    # The template emits JSON literals inside Pythonic keyword arguments. Accept
    # both JSON and Python spellings, without evaluating expressions or calls.
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
        return node.value
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, ast.List):
        return [_literal(value) for value in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            key = _literal(key)
            if not isinstance(key, str) or key in result:
                raise ValueError("Tool arguments require distinct string keys")
            result[key] = _literal(value)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if type(value) in (int, float):
            return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("Tool arguments must contain JSON values only")


@lru_cache(maxsize=256)
def _schema_validator(serialized):
    schema = json.loads(serialized)
    cls = validators.validator_for(schema)
    try:
        cls.check_schema(schema)
    except SchemaError as error:
        raise ValueError("Invalid tool parameter schema") from error
    return cls(schema, registry=Registry())


def _parameter_type_matches(value, schema, resolver, checker, seen):
    """Check wire types only; constraints belong to the task's reward policy."""
    if isinstance(schema, bool):
        return schema
    if id(schema) in seen:
        return True
    seen = seen | {id(schema)}
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(checker.is_type(value, kind) for kind in kinds):
            return False
    for key in ("$ref", "$dynamicRef"):
        if key in schema:
            resolved = resolver.lookup(schema[key])
            if not _parameter_type_matches(value, resolved.contents, resolved.resolver, checker, seen):
                return False
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            matches = [
                _parameter_type_matches(
                    value,
                    child,
                    resolver.in_subresource(Resource.from_contents(child, default_specification=DRAFT202012)),
                    checker,
                    seen,
                )
                for child in schema[key]
            ]
            # Branch exclusivity depends on full constraints, not just wire types.
            if not (all(matches) if key == "allOf" else any(matches)):
                return False
    return True


def parameter_value(value, schema, root_schema=None):
    """Decode native unquoted strings and JSON scalars/containers using the tool schema."""
    value = value.strip()
    root_schema = root_schema if root_schema is not None else schema
    validator = _schema_validator(json.dumps(root_schema, sort_keys=True, allow_nan=False))
    if schema is False:
        raise ValueError("Parameter forbidden by its schema")
    schema = schema if isinstance(schema, dict) else {}
    # Unconstrained and plain string parameters retain the native unquoted spelling.
    if not schema or schema.get("type") == "string":
        return value
    validator = validator.evolve(schema=schema)

    def matches_type(candidate):
        resource = Resource.from_contents(root_schema, default_specification=DRAFT202012)
        uri = resource.id() or ""
        registry = Registry().with_resource(uri, resource).crawl()
        scope = registry.resolver(uri)
        if schema is not root_schema:
            scope = scope.in_subresource(Resource.from_contents(schema, default_specification=DRAFT202012))
        return _parameter_type_matches(candidate, schema, scope, validator.TYPE_CHECKER, set())

    return _decode_parameter(value, validator.is_valid, matches_type)


def _decode_parameter(value, is_valid, matches_type):
    try:
        decoded = json.loads(value, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except json.JSONDecodeError:
        decoded = value
    json.dumps(decoded, allow_nan=False)
    try:
        if is_valid(decoded):
            return decoded
        if is_valid(value):
            return value
        # Preserve parseable model mistakes for comparator partial credit. The
        # structured-output verifier still validates the complete final schema.
        if matches_type(decoded):
            return decoded
        if matches_type(value):
            return value
    except Unresolvable as error:
        raise ValueError("Parameter schema references an unavailable resource") from error
    raise ValueError("Parameter value does not match its declared type/schema")


def property_value(value, name, schema):
    """Decode an object property through root references/composition in its own scope.

    Projection considers only this property's constraints. Required sibling fields
    and oneOf branch exclusivity are checked against the complete final instance.
    """
    _schema_validator(json.dumps(schema, sort_keys=True, allow_nan=False))
    if not any(key in schema for key in ("$ref", "$dynamicRef", "allOf", "anyOf", "oneOf")):
        return parameter_value(value, schema.get("properties", {}).get(name, {}), schema)
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    uri = resource.id() or ""
    registry = Registry().with_resource(uri, resource).crawl()
    validator = validators.validator_for(schema)(schema, registry=registry)

    def project(node, resolver, seen, types_only=False):
        if isinstance(node, bool):
            return (lambda candidate: False) if not node else None
        if id(node) in seen:
            return None
        seen = seen | {id(node)}
        checks = []
        if name in node.get("properties", {}):
            child = node["properties"][name]
            scope = resolver.in_subresource(Resource.from_contents(child, default_specification=DRAFT202012))
            if types_only:
                checks.append(
                    lambda candidate, child=child, scope=scope: _parameter_type_matches(
                        candidate, child, scope, validator.TYPE_CHECKER, set()
                    )
                )
            else:
                checks.append(
                    lambda candidate, child=child, scope=scope: not any(
                        validator.descend(candidate, child, resolver=scope)
                    )
                )
        for keyword in ("$ref", "$dynamicRef"):
            if keyword in node:
                resolved = resolver.lookup(node[keyword])
                checks.append(project(resolved.contents, resolved.resolver, seen, types_only))
        for child in node.get("allOf", []):
            scope = resolver.in_subresource(Resource.from_contents(child, default_specification=DRAFT202012))
            checks.append(project(child, scope, seen, types_only))
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                alternatives = [
                    project(
                        child,
                        resolver.in_subresource(Resource.from_contents(child, default_specification=DRAFT202012)),
                        seen,
                        types_only,
                    )
                    for child in node[keyword]
                ]
                checks.append(
                    lambda candidate, alternatives=alternatives: any(
                        check is None or check(candidate) for check in alternatives
                    )
                )
        checks = [check for check in checks if check is not None]
        return (lambda candidate: all(check(candidate) for check in checks)) if checks else None

    try:
        check = project(schema, registry.resolver(uri), set())
        if check is None:
            return value.strip()

        def matches_type(candidate):
            type_check = project(schema, registry.resolver(uri), set(), types_only=True)
            return type_check is None or type_check(candidate)

        return _decode_parameter(value.strip(), check, matches_type)
    except Unresolvable as error:
        raise ValueError("Parameter schema references an unavailable resource") from error


def parse_native_calls(body, tools):
    """Parse complete Olmo XML-style calls without a vLLM/SGLang dependency.

    This is the offline reward's strict parser, not a permissive serving parser.
    All non-whitespace action text must belong to a call. Parallel calls have
    separate wrappers, and duplicate parameters or malformed tails fail closed.
    """
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    calls = []
    remaining = body.strip()
    while remaining:
        match = re.match(r"<tool_call>\s*<function=([^<>]+)>(.*?)</function>\s*</tool_call>", remaining, re.DOTALL)
        if not match:
            raise ValueError("Expected a complete native function call")
        name, parameters = match[1], {}
        content = match[2].strip()
        while content:
            parameter = re.match(r"<parameter=([^<>]+)>(.*?)</parameter>", content, re.DOTALL)
            if not parameter or parameter[1] in parameters:
                raise ValueError("Malformed or duplicate parameter")
            key, value = parameter[1], parameter[2]
            if re.search(r"</?(?:tool_call|function|parameter)(?:[=>])", value):
                raise ValueError("Ambiguous structural tag in parameter")
            schema = schemas.get(name, {})
            parameters[key] = property_value(value, key, schema)
            content = content[parameter.end() :].strip()
        calls.append({"name": name, "parameters": parameters})
        remaining = remaining[match.end() :].strip()
    if not calls:
        raise ValueError("Empty tool call")
    return calls


def _boundary(text, delimiter):
    """Locate a delimiter outside literal strings and containers."""
    stack, quote, escaped = [], None, False
    pairs = {"[": "]", "{": "}", "(": ")"}
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == delimiter and not stack:
            return index
        elif char in ("'", '"'):
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "]})" and (not stack or stack.pop() != char):
            raise ValueError("Unbalanced call arguments")
    if stack or quote:
        raise ValueError("Unclosed call arguments")
    return len(text)


def parse_pythonic_calls(body: str, tools: Sequence[dict[str, Any]] = ()):
    """Read the template's literal call notation, including non-Python tool names.

    Names are matched against declarations before looking for parentheses: some
    ToolRL names themselves contain spaces, digits or parentheses. Values use a
    restricted literal parser, never Python evaluation or attribute resolution.
    """
    match = re.fullmatch(r"\s*<function_calls>(.*?)</function_calls>\s*", body, re.S)
    if match is None:
        raise ValueError("Expected a complete Pythonic function-call block")
    names: list[str] = [t["function"]["name"] for t in tools]
    names.sort(key=len, reverse=True)
    remaining, calls = match[1].strip(), []
    while remaining:
        name = next((n for n in names if remaining.startswith(n + "(")), None)
        if name is None:
            opening = remaining.find("(")
            if opening < 1 or not remaining[:opening].isidentifier():
                raise ValueError("Expected a declared tool name or function identifier")
            name = remaining[:opening]
        content = remaining[len(name) + 1 :]
        closing = _boundary(content, ")")
        if closing == len(content):
            raise ValueError("Unclosed function call")
        args, arguments = content[:closing].strip(), {}
        while args:
            end = _boundary(args, ",")
            argument = args[:end].strip()
            if argument.startswith("**"):
                values = _literal(ast.parse(argument[2:], mode="eval").body)
            else:
                key, separator, value = argument.partition("=")
                key = key.strip()
                if not separator or not key or any(c in key for c in "()[]{}<>\n"):
                    raise ValueError("Expected named literal arguments")
                values = {key: _literal(ast.parse(value.strip(), mode="eval").body)}
            if not isinstance(values, dict) or any(k in arguments for k in values):
                raise ValueError("Duplicate or malformed arguments")
            arguments.update(values)
            args = args[end + 1 :].strip()
        json.dumps(arguments, allow_nan=False)
        calls.append({"name": name, "parameters": arguments})
        remaining = content[closing + 1 :].strip()
        if remaining.startswith(";"):
            remaining = remaining[1:].strip()
    if not calls:
        raise ValueError("Empty function-call block")
    return calls
