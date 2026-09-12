"""Compile the qualified packed SGLang judge service from prepared immutable weights."""

import hashlib
import json
import sys
from pathlib import Path


def command(service, port):
    prepared = json.loads((Path(service["prepared_dir"]) / "prepared.json").read_text())
    if prepared.get("verdict") != "passed" or any(prepared.get(key) != service[key] for key in ("model", "revision")):
        raise ValueError("Prepared judge identity differs from the requested model/revision")
    if hashlib.sha256(Path(prepared["template"]).read_bytes()).hexdigest() != prepared["template_sha256"]:
        raise ValueError("Prepared judge template hash mismatch")
    if "<think>\n\n</think>" not in prepared.get("rendered_canary", ""):
        raise ValueError("Prepared judge template does not close its thinking block")
    model_config = json.loads((Path(prepared["snapshot"]) / "config.json").read_text())
    if service["max_context_length"] > model_config["max_position_embeddings"]:
        raise ValueError("Judge context exceeds the native model capacity")
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        prepared["snapshot"],
        "--served-model-name",
        service["model"],
        "--dtype",
        "bfloat16",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp-size",
        str(service["tensor_parallel_size"]),
        "--context-length",
        str(service["max_context_length"]),
        "--disable-cuda-graph",
        "--chat-template",
        prepared["template"],
        "--max-running-requests",
        str(service["max_concurrent_calls"]),
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.85",
    ]
