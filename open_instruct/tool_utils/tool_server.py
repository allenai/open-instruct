"""
This script sets up a FastAPI server that allows users to execute Python code snippets

python open_instruct/tool_utils/tool_server.py

```bash
docker build -t tool-server -f open_instruct/tool_utils/Dockerfile .
# Ai2 build:
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/tool-server
beaker image create tool-server -n tool-server -w ai2/$beaker_user

# Run the server
docker run -p 1212:1212 tool-server

# Ai2 run:
python mason.py \
    --cluster ai2/phobos-cirrascale \
    --workspace ai2/scaling-rl \
    --image $beaker_user/tool-server --pure_docker_mode \
    --priority high \
    --budget ai2/oe-adapt \
    --gpus 0 -- python tool_server.py
# https://beaker.org/ex/01JSSTZG7111M8T1CA2D9S662N
```


You can test with the following. You prob want to test to make sure the

1) the timeout works
2) the timeout in the first curl does not block the second curl

```
curl -X POST http://localhost:1212/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import time;time.sleep(4)", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'


curl -X POST http://localhost:1212/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "print(1)", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'

curl -X POST http://localhost:1212/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import sympy", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'

curl -X POST http://localhost:1212/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import sympy", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'
```

"""

import io
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
from typing import Optional
import asyncio
import concurrent.futures
import signal
import os
import threading

app = FastAPI(title="Python Code Executor API")

# Process pool and lock for safe recreation
process_pool = None
pool_lock = threading.Lock()

def get_process_pool():
    """Get the current process pool or create a new one if necessary."""
    global process_pool
    with pool_lock:
        if process_pool is None:
            process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=5)
        return process_pool

def reset_process_pool():
    """Safely reset the process pool when it breaks."""
    global process_pool
    with pool_lock:
        if process_pool is not None:
            process_pool.shutdown(wait=False)
            process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=5)
        return process_pool

class CodeRequest(BaseModel):
    code: str
    timeout: Optional[int] = 5  # Default timeout in seconds

class CodeResponse(BaseModel):
    output: str
    error: Optional[str] = None
    success: bool

def run_code_in_process(code, result_queue):
    """Execute code in a separate process and put results in queue."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    try:
        # Create a shared global namespace for the executed code
        # This is the key fix - providing both globals and locals to exec
        global_namespace = {}
        
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # Execute with explicit global namespace so recursive functions work
            exec(code, global_namespace)
        
        output = stdout.getvalue()
        error = stderr.getvalue()
        
        result = {
            "output": output,
            "error": error if error else None,
            "success": True
        }
        
    except Exception as e:
        error_message = f"Error executing code: {str(e)}\n"
        error_traceback = traceback.format_exc()
        result = {
            "output": "",
            "error": error_message + error_traceback,
            "success": False
        }
    
    result_queue.put(result)

def execute_with_timeout(code, timeout):
    """Execute code with timeout and return result."""
    # Create a queue to get the result from the process
    result_queue = multiprocessing.Queue()
    
    # Create and start the process
    process = multiprocessing.Process(
        target=run_code_in_process, 
        args=(code, result_queue)
    )
    process.start()
    
    # Wait for the process to finish with timeout
    process.join(timeout=timeout)
    
    # If process is still alive after timeout, terminate it
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)  # Give it a second to terminate
        
        # Force kill if still alive (uncommon but possible)
        if process.is_alive():
            try:
                os.kill(process.pid, signal.SIGKILL)
            except:
                pass  # Process might have terminated between check and kill
            process.join(timeout=1)
        
        return {
            "output": "",
            "error": f"Execution timed out after {timeout} seconds",
            "success": False
        }
    
    # Process completed within timeout, get the result
    if not result_queue.empty():
        return result_queue.get()
    else:
        return {
            "output": "",
            "error": "Execution completed but produced no result",
            "success": False
        }

@app.post("/execute", response_model=CodeResponse)
async def execute_code(request: CodeRequest):
    """
    Execute a Python code snippet with timeout and return the output or error.
    Uses a process pool for better performance while maintaining isolation.
    """
    loop = asyncio.get_event_loop()
    
    # Get the current process pool
    pool = get_process_pool()
    
    try:
        # Execute in process pool with timeout
        result = await loop.run_in_executor(
            pool,
            execute_with_timeout,
            request.code,
            request.timeout
        )
        return CodeResponse(**result)
    except concurrent.futures.BrokenProcessPool:
        # Handle broken pool by recreating it
        pool = reset_process_pool()
        
        try:
            # Retry once with the new pool
            result = await loop.run_in_executor(
                pool,
                execute_with_timeout,
                request.code,
                request.timeout
            )
            return CodeResponse(**result)
        except Exception as e:
            # If retry fails, return an error
            return CodeResponse(
                output="",
                error=f"Server error during code execution: {str(e)}",
                success=False
            )
    except Exception as e:
        # Handle other exceptions
        return CodeResponse(
            output="",
            error=f"Error executing code: {str(e)}",
            success=False
        )

@app.get("/")
async def root():
    return {"message": "Python Code Executor API. Use /execute endpoint to run Python code."}

if __name__ == "__main__":
    import uvicorn
    # Initialize the process pool
    get_process_pool()
    uvicorn.run(app, host="0.0.0.0", port=1212)