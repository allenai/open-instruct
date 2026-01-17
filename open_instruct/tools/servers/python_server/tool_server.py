"""
# https://chatgpt.com/share/681146aa-d850-8011-a959-19a2c1caee5d, tldr: performance optimization by o3

This script sets up a FastAPI server that allows users to execute Python code snippets

cd open_instruct/tool_utils
PREIMPORT_PKGS=pandas,numpy,sympy,time,math,networkx uv run uvicorn tool_server:app --host 0.0.0.0 --port 1212

```bash
docker build -t tool-server .
# Ai2 build:
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/tool-server
beaker image create tool-server -n tool-server -w ai2/$beaker_user
# ghcr build:
docker build -t ghcr.io/allenai/open-instruct/python-code-executor -f open_instruct/tool_utils/Dockerfile .
docker push ghcr.io/allenai/open-instruct/python-code-executor

# Run the server
docker run -p 1212:8080 tool-server

# gcloud run deploy:
gcloud run deploy open-instruct-tool-server --project ai2-allennlp --region us-central1 --source .

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
curl -X POST https://open-instruct-tool-server-10554368204.us-central1.run.app/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import time;time.sleep(4)", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'


curl -X POST https://open-instruct-tool-server-10554368204.us-central1.run.app/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "print(1)", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'

curl -X POST https://open-instruct-tool-server-10554368204.us-central1.run.app/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import sympy", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'

curl -X POST https://open-instruct-tool-server-10554368204.us-central1.run.app/execute \
     -H "Content-Type: application/json" \
     -d '{"code": "import sympy", "timeout": 3}' \
     -w '\nTotal time: %{time_total}s\n'
```

"""

###############################################################################
# Imports & global config
###############################################################################
import ast
import asyncio
import importlib
import io
import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stderr, redirect_stdout, suppress

from fastapi import FastAPI
from pydantic import BaseModel

from open_instruct import logger_utils

###############################################################################
# Configure logging (Cloud Run uses stdout/stderr)
###############################################################################
logger = logger_utils.setup_logger(__name__)

###############################################################################
# Pre‑import heavy libraries *before* the fork so children share memory
###############################################################################
DEFAULT_PKGS = ["pandas", "numpy", "sympy", "time", "math", "networkx"]
PREIMPORT_PKGS = [pkg.strip() for pkg in os.getenv("PREIMPORT_PKGS", ",".join(DEFAULT_PKGS)).split(",") if pkg.strip()]

for _pkg in PREIMPORT_PKGS:
    try:
        importlib.import_module(_pkg)
        logger.info("Pre‑imported package: %s", _pkg)
    except ModuleNotFoundError:
        logger.warning("Package not found for pre‑import: %s", _pkg)

###############################################################################
# Pool initialiser (runs in each worker process)
###############################################################################


def _worker_init():  # noqa: D401
    """Run in every worker process to ensure packages are imported."""
    for _pkg in PREIMPORT_PKGS:
        with suppress(ModuleNotFoundError):
            importlib.import_module(_pkg)


###############################################################################
# Create a single, long‑lived pool
###############################################################################
POOL_SIZE = int(os.getenv("POOL_SIZE", os.cpu_count() or 1))
process_pool: ProcessPoolExecutor | None = ProcessPoolExecutor(max_workers=POOL_SIZE, initializer=_worker_init)
logger.info("Process pool started with %s workers", POOL_SIZE)

app = FastAPI(title="Python Code Executor (Optimised, Logged)")


###############################################################################
# REPL‑style transformation
###############################################################################
class _ReplPrinter(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            return node
        return ast.copy_location(
            ast.Expr(
                ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=[node.value],
                    keywords=[ast.keyword(arg="flush", value=ast.Constant(True))],
                )
            ),
            node,
        )


def _compile_repl(code: str):
    try:
        tree = ast.parse(code, mode="exec")
        tree = _ReplPrinter().visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, "<user-snippet>", "exec")
    except SyntaxError:
        return code


###############################################################################
# Worker function (executes user code)
###############################################################################


def _run_user_code(code: str, timeout: int = 5):
    def _timeout_handler(signum, frame):  # noqa: D401, ANN001, ARG001
        raise TimeoutError("execution timed out")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    stdout, stderr = io.StringIO(), io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            # 👉 compile() directly without REPL transformation
            compiled = compile(code, "<user-snippet>", "exec")
            exec(compiled, {})  # noqa: S102
        result = {"output": stdout.getvalue(), "error": stderr.getvalue() or None, "success": True}
    except Exception as exc:  # pylint: disable=broad-except
        result = {"output": "", "error": f"{exc}\n{traceback.format_exc()}", "success": False}
    finally:
        signal.alarm(0)

    return result


###############################################################################
# FastAPI schema
###############################################################################
class CodeRequest(BaseModel):
    code: str
    timeout: int | None = 5


class CodeResponse(BaseModel):
    output: str
    error: str | None = None
    success: bool


###############################################################################
# Endpoints
###############################################################################
@app.post("/execute", response_model=CodeResponse)
async def execute_code(req: CodeRequest):  # noqa: D401
    global process_pool  # noqa: PLW0603

    # Log input (truncate to 200 chars to avoid huge logs)
    truncated_code = (req.code[:197] + "...") if len(req.code) > 200 else req.code
    logger.info("📥 Received request (timeout=%s s): \n%s", req.timeout, truncated_code)

    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(process_pool, _run_user_code, req.code, req.timeout)

    try:
        result = await asyncio.wait_for(fut, timeout=req.timeout + 1)
    except asyncio.TimeoutError:
        fut.cancel()
        result = {"output": "", "error": f"Execution timed out after {req.timeout} seconds (outer)", "success": False}
    except BrokenProcessPool:
        logger.error("Pool broken — recreating")
        process_pool = ProcessPoolExecutor(max_workers=POOL_SIZE, initializer=_worker_init)
        result = {"output": "", "error": "Worker pool crashed and was restarted. Please retry.", "success": False}

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "📤 Responding success=%s latency=%.1f ms output_len=%s error=%s, output=\n%s",
        result["success"],
        latency_ms,
        len(result["output"] or ""),
        bool(result["error"]),
        result["output"],
    )
    if result["error"] is not None:
        logger.debug("Error: %s", result["error"])

    return CodeResponse(**result)


@app.get("/")
async def root():  # noqa: D401
    return {"message": "Python Code Executor API — POST /execute {code, timeout}"}
