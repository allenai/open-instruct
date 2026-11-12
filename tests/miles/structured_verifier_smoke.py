"""Exercise structured-grader imports and schema checks during image builds."""

from open_instruct.miles.rewards import structured_outputs


def main():
    verifier = structured_outputs.StructuredOutputVerifier()
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    cases = {"json": '{"n": 3}', "yaml": "n: 3", "toml": "n = 3", "xml": "<n>3</n>", "csv": "n\n3"}
    for kind, answer in cases.items():
        target = {
            "format": kind,
            "schema_policy": "as_declared",
            "coerce_types": True,
            "schema": {"type": "array", "items": schema} if kind == "csv" else schema,
        }
        assert verifier([], answer, target).score == 1.0, f"Valid {kind} answer rejected"
        assert verifier([], answer.replace("3", "bad"), target).score == 0.0, f"Invalid {kind} answer accepted"


if __name__ == "__main__":
    main()
