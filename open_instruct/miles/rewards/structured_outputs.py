"""Versioned structured-format contracts shared by screening and training.

Schema strengthening and type coercion are explicit target policies. Evaluation
runs in the existing bounded verifier workers, with a per-response alarm covering
parsing, reference resolution and every regex performed by the schema library.
"""

import copy
import csv
import io
import json
import os
import signal
from dataclasses import dataclass

import tomllib
import xmltodict
import yaml
from openapi_schema_validator import OAS31Validator
from referencing import Registry

from open_instruct.miles.rewards import candidate_verifiers as common
from open_instruct.miles.rewards import nemo_structured_types, rewards

FORMATS = {"json", "yaml", "toml", "xml", "csv"}


def validate_target(target):
    if not isinstance(target, dict) or target.get("format") not in FORMATS:
        raise ValueError("Structured target requires format=json/yaml/toml/xml/csv")
    if set(target) - {"format", "schema", "schema_policy", "coerce_types"}:
        raise ValueError("Unknown structured target fields")
    if target.get("schema_policy") not in {"as_declared", "nemo_strict_v1"}:
        raise ValueError("Specify schema_policy explicitly: as_declared or nemo_strict_v1")
    if type(target.get("coerce_types", False)) is not bool:
        raise ValueError("coerce_types must be boolean")
    schema = copy.deepcopy(target["schema"])
    if not isinstance(schema, dict):
        raise ValueError("Expected schema object")
    if target["schema_policy"] == "nemo_strict_v1":
        nemo_structured_types.NativeStructuredTypes().strictify_schema(schema)
    # Only schema positions are inspected: a field literally called '$ref' is fine.
    pending = [schema]
    maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
    singles = {
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "additionalItems",
        "unevaluatedItems",
    }
    lists = {"allOf", "anyOf", "oneOf", "prefixItems"}
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        for key in ("$ref", "$dynamicRef"):
            if key in node and (not isinstance(node[key], str) or not node[key].startswith("#")):
                raise ValueError("Only local schema references are supported")
        if "$id" in node and node is not schema:
            raise ValueError("Nested resource identifiers require a separately qualified contract")
        for key, value in node.items():
            if key in maps and isinstance(value, dict):
                pending.extend(value.values())
            elif key in singles:
                pending.extend(value if isinstance(value, list) else [value])
            elif key in lists and isinstance(value, list):
                pending.extend(value)
            elif key == "dependencies" and isinstance(value, dict):
                pending.extend(v for v in value.values() if isinstance(v, dict))
    OAS31Validator.check_schema(schema)
    return schema


class UniqueYAMLLoader(yaml.SafeLoader):
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
    # Reject nonfinite values, cyclic aliases and non-JSON scalar types explicitly.
    json.dumps(value, allow_nan=False)
    return value


@dataclass
class StructuredConfig:
    timeout_seconds: float = 1.0


class StructuredOutputVerifier:
    get_config_class = staticmethod(lambda: StructuredConfig)

    def __init__(self, verifier_config=None):
        self.config = verifier_config or StructuredConfig()
        if not 0 < self.config.timeout_seconds <= 30:
            raise ValueError("Structured verifier timeout must be in (0, 30] seconds")

    async def async_call(self, tokenized_prediction, prediction, label, **kwargs):
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
        return result

    def __call__(self, tokenized_prediction, prediction, label, **kwargs):
        # This synchronous entry point executes on the verifier worker's main thread.
        def expired(signum, frame):
            raise TimeoutError("Structured response exceeded its verification budget")

        policy = os.environ.get(
            "OI_MILES_FORMAT_TIMEOUT_POLICY", os.environ.get("OI_MILES_MATH_TIMEOUT_POLICY", "zero")
        )
        if policy not in {"zero", "raise"}:
            raise ValueError("Structured timeout policy must be zero or raise")
        previous = signal.signal(signal.SIGALRM, expired)
        timer = signal.setitimer(signal.ITIMER_REAL, self.config.timeout_seconds)
        try:
            schema = validate_target(label)  # Configuration errors must not become wrong answers.
            text = common.answer_text(prediction)
            try:
                value = parse(text, label["format"], schema, label.get("coerce_types", False))
            except (
                ValueError,
                TypeError,
                RecursionError,
                yaml.YAMLError,
                csv.Error,
                xmltodict.expat.ExpatError,
            ) as error:
                return common.Result(
                    0.0, diagnostics={"kind": "structured_v1", "status": "parse_error", "error": str(error)}
                )
            # An explicit empty Registry refuses retrieval by default, including
            # network access. Do not use jsonschema's legacy default registry.
            validator = OAS31Validator(schema, registry=Registry())
            return common.Result(
                float(validator.is_valid(value)), diagnostics={"kind": "structured_v1", "status": "completed"}
            )
        except TimeoutError as error:
            if policy == "raise":
                raise
            return common.Result(0.0, diagnostics={"kind": "structured_v1", "status": "timeout", "error": str(error)})
        finally:
            signal.setitimer(signal.ITIMER_REAL, *timer)
            signal.signal(signal.SIGALRM, previous)
