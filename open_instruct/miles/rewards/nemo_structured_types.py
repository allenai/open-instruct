# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers from NVIDIA NeMo Gym structured_outputs/app.py.

Source SHA256: 5e9d85edaa6743abcdf0fd80859321bb9ce386e461846399069e80bb25cbbc70.
Adapted to leave boolean CSV subschemas to validation and to strengthen only
schema positions, including combinators. See docs/miles/dataset-verifiers.md.
"""

from typing import Any


def schema_nodes(schema):
    """Walk schema objects without treating property maps or instance defaults as schemas."""
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
        yield node
        for key, value in node.items():
            if key in maps and isinstance(value, dict):
                pending.extend(value.values())
            elif key in singles:
                pending.extend(value if isinstance(value, list) else [value])
            elif key in lists and isinstance(value, list):
                pending.extend(value)
            elif key == "dependencies" and isinstance(value, dict):
                pending.extend(v for v in value.values() if isinstance(v, dict))


class NativeStructuredTypes:
    def strictify_schema(self, schema: dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        for node in schema_nodes(schema):
            if isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False

    def coerce_xml_types(self, data: Any, schema: dict[str, Any]) -> Any:
        """Recursively coerce xmltodict string values to match the JSON schema types.

        xmltodict.parse() returns all leaf values as strings. This method walks the
        parsed data alongside the schema and converts values where possible.
        On conversion failure the original value is returned so that schema
        validation can report the error.
        """
        if not isinstance(schema, dict) or "type" not in schema:
            return data

        schema_type = schema["type"]

        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            coerced = {}
            for key, value in data.items():
                if key in properties:
                    coerced[key] = self.coerce_xml_types(value, properties[key])
                else:
                    coerced[key] = value
            return coerced

        if schema_type == "array":
            items_schema = schema.get("items", {})
            # xmltodict represents repeated child elements as {"tagName": [values]},
            # e.g. <skills><string>a</string><string>b</string></skills> becomes
            # {"string": ["a", "b"]}. For single elements, xmltodict gives
            # {"string": "python"} instead of a list. In both cases, unwrap the
            # single-key dict since we're at an array schema position -- a dict here
            # is always the xmltodict wrapping artifact, not a meaningful structure.
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if not isinstance(data, list):
                data = [data] if data is not None else []
            return [self.coerce_xml_types(item, items_schema) for item in data]

        # xmltodict returns None for empty tags like <field/> or <field></field>.
        # Coerce to "" only for string types (parity with JSON/YAML where "" is valid).
        # Non-string types (integer, boolean, etc.) intentionally left as None so
        # they fail validation -- 0 and False are meaningful values, not "empty".
        if data is None and schema_type == "string":
            return ""

        if isinstance(data, str):
            try:
                if schema_type == "integer":
                    return int(data)
                if schema_type == "number":
                    return float(data)
                if schema_type == "boolean":
                    lower = data.lower()
                    if lower in ("true", "1"):
                        return True
                    if lower in ("false", "0"):
                        return False
            except (ValueError, AttributeError):
                pass

        return data

    def coerce_csv_types(self, rows: list[dict[str, str]], schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Coerce CSV string values to match JSON Schema types.

        csv.DictReader returns all values as strings.  Walk rows alongside the
        items schema and convert values where possible.  On conversion failure
        the original string is kept so that schema validation reports the error.
        """
        items_schema = schema.get("items", schema)
        if not isinstance(items_schema, dict):
            return rows
        properties = items_schema.get("properties", {})

        coerced_rows = []
        for row in rows:
            coerced: dict[str, Any] = {}
            for key, value in row.items():
                prop_schema = properties.get(key, {})
                prop_type = prop_schema.get("type", "string") if isinstance(prop_schema, dict) else "string"
                coerced[key] = self._coerce_csv_scalar(value, prop_type)
            coerced_rows.append(coerced)
        return coerced_rows

    def _coerce_csv_scalar(self, value: str, target_type) -> Any:
        """Coerce a single CSV string to the target JSON Schema type."""
        if isinstance(target_type, list):
            if (value is None or value == "") and "null" in target_type:
                return None
            for t in target_type:
                if t == "null":
                    continue
                result = self._coerce_csv_scalar(value, t)
                if not isinstance(result, str) or t == "string":
                    return result
            return value

        if value is None or value == "":
            return value

        try:
            if target_type == "integer":
                return int(value)
            if target_type == "number":
                return float(value)
            if target_type == "boolean":
                lower = value.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except (ValueError, AttributeError):
            pass
        return value
