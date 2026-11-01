"""Schema-only reward for prepared JSON-output slices (no factual grading)."""

import importlib

from open_instruct.miles.rewards import candidate_verifiers, task_verifiers


def validate_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Expected a JSON schema object")
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            # Preparation resolves/qualifies local references; runtime never fetches a URL.
            if any(
                not isinstance(v, str) or not v.startswith("#")
                for k, v in node.items()
                if k in {"$ref", "$dynamicRef"}
            ):
                raise ValueError("Only local schema references are allowed")
            if "$id" in node:
                raise ValueError("Nested/resource schema identifiers are not supported")
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    validator = importlib.import_module("openapi_schema_validator").OAS31Validator
    validator.check_schema(schema)
    return validator(schema)


class JSONSchemaVerifier:
    get_config_class = staticmethod(lambda: task_verifiers.RewardConfig)

    def __init__(self, verifier_config=None):
        pass

    async def async_call(self, tokens, prediction, label, **kwargs):
        validator = validate_schema(label)
        try:
            value = candidate_verifiers.strict_json(candidate_verifiers.answer_text(prediction))
        except (ValueError, RecursionError):
            return candidate_verifiers.Result(0.0)
        # Schema resolution failures propagate; only instance failures are reward zero.
        return candidate_verifiers.Result(float(validator.is_valid(value)))
