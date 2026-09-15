"""Prepare immutable teacher weights and exec a supervised SGLang scoring server."""

import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def command(service, port):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        service["snapshot"],
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tp",
        str(service["tensor_parallel_size"]),
        "--mem-fraction-static",
        "0.6",
        "--context-length",
        str(service["max_context_length"]),
        "--max-running-requests",
        str(service["concurrency"]),
        "--max-total-tokens",
        str(service["max_context_length"] * service["concurrency"]),
        "--attention-backend",
        service["attention_backend"],
        "--sampling-backend",
        "pytorch",
        "--disable-flashinfer-autotune",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]


def main():
    path, port = Path(sys.argv[1]), int(sys.argv[2])
    service = json.loads(path.read_text())
    source = service["source"]
    snapshot = source if Path(source).is_absolute() else snapshot_download(source, revision=service["revision"])
    service["snapshot"] = snapshot
    config = Path(snapshot) / "config.json"
    service["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    service["model_type"] = json.loads(config.read_text())["model_type"]
    path.write_text(json.dumps(service, indent=2))
    argv = command(service, port)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
