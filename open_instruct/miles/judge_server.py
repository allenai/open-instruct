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
    extension = service.get("context_extension", "none")
    overrides = []
    if extension == "qwen3-yarn-128k":
        if service["model"] != "Qwen/Qwen3-32B" or model_config.get("model_type") != "qwen3":
            raise ValueError("qwen3-yarn-128k requires the Qwen/Qwen3-32B checkpoint")
        if model_config.get("rope_scaling") or service["max_context_length"] > 131072:
            raise ValueError("qwen3-yarn-128k requires an unscaled checkpoint and context at most 131072")
        overrides = [
            "--json-model-override-args",
            json.dumps(
                {"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
            ),
        ]
    elif extension != "none":
        raise ValueError(f"Unknown judge context extension: {extension}")
    elif service["max_context_length"] > model_config["max_position_embeddings"]:
        raise ValueError("Judge context exceeds the native model capacity; configure a supported context_extension")
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
        *overrides,
    ]
