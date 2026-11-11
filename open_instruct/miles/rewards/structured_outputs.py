"""Versioned structured-format contracts shared by screening and training.

Schema strengthening and type coercion are explicit target policies. Evaluation
runs in the existing bounded verifier workers, with a per-response alarm covering
parsing, reference resolution and every regex performed by the schema library.
"""

import copy
import csv
import io
import json
import signal
import time
from dataclasses import dataclass

import tomllib
import xmltodict
import yaml
from openapi_schema_validator import OAS31Validator
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from open_instruct import logger_utils
from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import nemo_structured_types, rewards, structured_tool_calls

FORMATS = {"json", "yaml", "toml", "xml", "csv"}
logger = logger_utils.setup_logger(__name__)


def validate_target(target):
    if not isinstance(target, dict) or target.get("format") not in FORMATS:
        raise ValueError("Structured target requires format=json/yaml/toml/xml/csv")
    if set(target) - {
        "format",
        "schema",
        "schema_policy",
        "coerce_types",
        "response_mode",
        "tool_name",
        "tool_payload_key",
        "tool_call_format",
    }:
        raise ValueError("Unknown structured target fields")
    if target.get("schema_policy") not in {"as_declared", "nemo_strict_v1"}:
        raise ValueError("Specify schema_policy explicitly: as_declared or nemo_strict_v1")
    if type(target.get("coerce_types", False)) is not bool:
        raise ValueError("coerce_types must be boolean")
    structured_tool_calls.validate_contract(target)
    schema = copy.deepcopy(target["schema"])
    if not isinstance(schema, dict):
        raise ValueError("Expected schema object")
    if target["schema_policy"] == "nemo_strict_v1":
        nemo_structured_types.NativeStructuredTypes().strictify_schema(schema)
    # Only schema positions are inspected: a field literally called '$ref' is fine.
    for node in nemo_structured_types.schema_nodes(schema):
        for key in ("$ref", "$dynamicRef"):
            if key in node and (not isinstance(node[key], str) or not node[key].startswith("#")):
                raise ValueError("Only local schema references are supported")
    OAS31Validator.check_schema(schema)
    # check_schema checks syntax, not reference resolution. Check every schema
    # resource in its own $id scope before accepting a source row, including
    # optional branches that a qualification control might never exercise.
    root = DRAFT202012.create_resource(schema)
    root_uri = root.id() or ""
    registry = Registry().with_resource(root_uri, root).crawl()
    pending = [(root, registry.resolver(root_uri))]
    while pending:
        resource, resolver = pending.pop()
        if isinstance(resource.contents, dict):
            for key in ("$ref", "$dynamicRef"):
                if key in resource.contents:
                    try:
                        resolver.lookup(resource.contents[key])
                    except Unresolvable as error:
                        raise ValueError(
                            f"Unresolvable schema reference {resource.contents[key]!r}: {type(error).__name__}"
                        ) from error
        pending.extend((child, resolver.in_subresource(child)) for child in resource.subresources())
    return schema


class UniqueYAMLLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key, value in node.value:
            key = self.construct_object(key, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings for JSON Schema validation")
            if key in mapping:
                raise ValueError("Duplicate YAML key")
            mapping[key] = self.construct_object(value, deep=deep)
        return mapping


def parse(text, kind, schema, coerce):
    native = nemo_structured_types.NativeStructuredTypes()
    if kind == "json":
        value = common.strict_json(text)
    elif kind == "yaml":
        value = yaml.load(text, Loader=UniqueYAMLLoader)
    elif kind == "toml":
        value = tomllib.loads(text)
    elif kind == "xml":
        if "<!DOCTYPE" in text or "<!ENTITY" in text:
            raise ValueError("XML declarations/entities are not supported")
        value = xmltodict.parse(text, disable_entities=True)
        if coerce:
            value = native.coerce_xml_types(value, schema)
    else:
        reader = csv.DictReader(io.StringIO(text), strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV requires distinct named columns")
        value = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in value):
            raise ValueError("CSV row width differs from its header")
        if coerce:
            value = native.coerce_csv_types(value, schema)
    # Reject nonfinite values and non-JSON scalars (dates, bytes, sets).
    # YAML aliases are rejected earlier, before construction or expansion.
    json.dumps(value, allow_nan=False)
    return value


@dataclass
class StructuredConfig:
    timeout_seconds: float = 1.0


def timeout_result(error, start):
    logger.warning("Structured verifier timeout; assigning zero reward: %s", error)
    return common.Result(
        0.0,
        diagnostics={
            "kind": "structured_v1",
            "status": "timeout",
            "error": str(error),
            "failure_policy": "zero",
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def instance_error_result(error, start):
    return common.Result(
        0.0,
        diagnostics={
            "kind": "structured_v1",
            "status": "validation_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "failure_policy": "zero",
            "elapsed_seconds": time.monotonic() - start,
        },
    )


class StructuredOutputVerifier:
    get_config_class = staticmethod(lambda: StructuredConfig)

    def __init__(self, verifier_config=None):
        self.config = verifier_config or StructuredConfig()
        if not 0 < self.config.timeout_seconds <= 30:
            raise ValueError("Structured verifier timeout must be in (0, 30] seconds")

    async def async_call(self, tokenized_prediction, prediction, label, **kwargs):
        start = time.monotonic()
        try:
            result = await rewards.isolated_verifier_call(
                {
                    "factory": "open_instruct.miles.rewards.structured_outputs.StructuredOutputVerifier",
                    "config": {"timeout_seconds": self.config.timeout_seconds},
                },
                tokenized_prediction,
                prediction,
                label,
                **kwargs,
            )
        except rewards.MathVerifierTimeout as error:
            return timeout_result(error, start)
        if result.diagnostics.get("status") == "timeout":
            # Worker stderr is suppressed by the pool; surface its timeout here.
            logger.warning("Structured verifier timeout; assigning zero reward: %s", result.diagnostics["error"])
        return result

    def __call__(self, tokenized_prediction, prediction, label, **kwargs):
        # This synchronous entry point executes on the verifier worker's main thread.
        def expired(signum, frame):
            raise TimeoutError("Structured response exceeded its verification budget")

        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, expired)
        timer = signal.setitimer(signal.ITIMER_REAL, self.config.timeout_seconds)
        try:
            try:
                schema = validate_target(label)  # Configuration errors must not become wrong answers.
                text = common.answer_text(prediction)
                try:
                    if label.get("response_mode") == "tool_call":
                        value = structured_tool_calls.parse_payload(text, label)
                    else:
                        value = parse(text, label["format"], schema, label.get("coerce_types", False))
                except (RecursionError, OverflowError) as error:
                    return instance_error_result(error, start)
                except (ValueError, TypeError, yaml.YAMLError, csv.Error, xmltodict.expat.ExpatError) as error:
                    return common.Result(
                        0.0, diagnostics={"kind": "structured_v1", "status": "parse_error", "error": str(error)}
                    )
                # An explicit empty Registry refuses retrieval by default, including
                # network access. Do not use jsonschema's legacy default registry.
                validator = OAS31Validator(schema, registry=Registry())
                try:
                    valid = validator.is_valid(value)
                except (RecursionError, OverflowError) as error:
                    # A parsed response can exceed the validator's recursion or
                    # numeric limits. Fail closed for this instance only; target
                    # checks and validator construction stay outside this guard.
                    return instance_error_result(error, start)
                return common.Result(float(valid), diagnostics={"kind": "structured_v1", "status": "completed"})
            finally:
                # Disarming can itself receive the pending alarm. Always restore
                # the handler/timer, then handle the timeout outside cleanup.
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, previous)
                    signal.setitimer(signal.ITIMER_REAL, *timer)
        except TimeoutError as error:
            return timeout_result(error, start)
