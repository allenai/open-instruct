"""
This script sets up a FastAPI server that allows users to execute Python code snippets

python open_instruct/tool_utils/tool_server.py


docker build -t tool-server -f open_instruct/tool_utils/Dockerfile .
docker run -p 1212:1212 tool-server
"""

import io
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

app = FastAPI(title="Python Code Executor API")

class CodeRequest(BaseModel):
    code: str
    timeout: Optional[int] = 5  # Default timeout in seconds

class CodeResponse(BaseModel):
    output: str
    error: Optional[str] = None
    success: bool

@app.post("/execute", response_model=CodeResponse)
async def execute_code(request: CodeRequest):
    """
    Execute a Python code snippet and return the output or error.
    """
    # Capture both stdout and stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    try:
        # Redirect both stdout and stderr
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # Execute the code
            exec(request.code)
        
        # Get the captured output
        output = stdout.getvalue()
        error = stderr.getvalue()
        
        # Combine output and error if both exist
        if error:
            if output:
                return CodeResponse(output=output, error=error, success=True)
            else:
                return CodeResponse(output="", error=error, success=True)
        else:
            return CodeResponse(output=output, success=True)
            
    except Exception as e:
        # Capture any exceptions that occur during execution
        error_message = f"Error executing code: {str(e)}\n"
        error_traceback = traceback.format_exc()
        return CodeResponse(output="", error=error_message + error_traceback, success=False)

@app.get("/")
async def root():
    return {"message": "Python Code Executor API. Use /execute endpoint to run Python code."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1212) 