"""
Manufactoria DSL verification API.

Launch locally with:
```
uv run uvicorn examples.manufactoria.api:app --host 0.0.0.0 --port 1235
```
"""

import re

from fastapi import FastAPI
from pydantic import BaseModel

from examples.manufactoria import parser as manufactoria_parser
from open_instruct import logger_utils

app = FastAPI()

logger = logger_utils.setup_logger(__name__)


class TestCase(BaseModel):
    input: str
    expected_output: str = ""
    expected_accepted: bool = True
    check_output: bool = True
    description: str = ""


class ManufactoriaTestRequest(BaseModel):
    dsl: str
    test_cases: list[TestCase]
    max_execution_time: float = 1.0


@app.post("/test_solution")
async def test_solution(request: ManufactoriaTestRequest):
    """Test a Manufactoria DSL solution against test cases."""
    try:
        factory = manufactoria_parser.create_robot_factory(request.dsl)
        results = []

        for test_case in request.test_cases:
            result = factory.process_robot(test_case.input)

            if test_case.check_output:
                has_regex_patterns = any(
                    char in test_case.expected_output for char in [".", "+", "*", "?", "|", "(", ")"]
                )
                if has_regex_patterns:
                    try:
                        output_matches = bool(re.fullmatch(test_case.expected_output, result.final_tape))
                    except re.error:
                        output_matches = result.final_tape == test_case.expected_output
                else:
                    output_matches = result.final_tape == test_case.expected_output
                passed = (output_matches and result.finished) == test_case.expected_accepted
            else:
                passed = result.finished == test_case.expected_accepted

            test_result = {
                "input": test_case.input,
                "expected_output": test_case.expected_output,
                "actual_output": result.final_tape,
                "expected_accepted": test_case.expected_accepted,
                "actual_accepted": result.finished,
                "check_output": test_case.check_output,
                "passed": passed,
                "path": result.path,
                "rejection_reason": result.rejection_reason,
                "description": test_case.description,
            }
            results.append(test_result)

        all_passed = all(result["passed"] for result in results)
        return {"valid": True, "all_passed": all_passed, "results": results}
    except manufactoria_parser.ParseError as e:
        logger.warning(f"Manufactoria DSL parse error: {e}")
        return {"valid": False, "all_passed": False, "message": f"DSL Parse Error: {e}", "results": []}
    except Exception as e:
        logger.warning(f"Manufactoria API error: {e}")
        return {"valid": False, "all_passed": False, "message": f"Unexpected error: {e}", "results": []}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import requests

    url = "http://localhost:1235/test_solution"

    payload = {
        "dsl": """
START start:
    NEXT end

END end
""",
        "test_cases": [
            {
                "input": "",
                "expected_output": "",
                "expected_accepted": True,
                "check_output": False,
                "description": "Empty input should be accepted",
            },
            {
                "input": "R",
                "expected_output": "",
                "expected_accepted": False,
                "check_output": False,
                "description": "Non-empty input should be rejected (no path to end)",
            },
        ],
        "max_execution_time": 1.0,
    }

    response = requests.post(url, json=payload)
    response_json = response.json()
    print("Status Code:", response.status_code)
    print("Response:", response_json)
    assert response_json["all_passed"] is True
    print("All tests passed.")
