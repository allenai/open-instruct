from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .code_utils import get_successful_tests_fast

app = FastAPI()

class TestRequest(BaseModel):
    program: str
    tests: List[str]
    max_execution_time: float = 1.0

@app.post("/test_program")
async def test_program(request: TestRequest):
    try:
        results = get_successful_tests_fast(
            program=request.program,
            tests=request.tests,
            max_execution_time=request.max_execution_time
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 