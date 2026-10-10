"""Serve a frozen same-tokenizer teacher with bounded scoring concurrency."""

import json
import os
import sys
from pathlib import Path
from urllib import request

from transformers import AutoTokenizer

from open_instruct.miles.distillation import alignment


def command(service, port):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        service["source"],
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tp",
        str(service["tensor_parallel_size"]),
        "--mem-fraction-static",
        "0.7",
        "--context-length",
        str(service["max_context_length"]),
        "--max-running-requests",
        str(service["concurrency"]),
        "--max-total-tokens",
        "131072",
        "--chunked-prefill-size",
        "8192",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--max-mamba-cache-size",
        "16",
    ]


def probe(path):
    service = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(service["source"], trust_remote_code=True)
    ids = tokenizer.encode("A small alignment check: 1 + 1 = 2", add_special_tokens=False)
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    req = request.Request(
        service["endpoint"] + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=service["request_timeout"]) as response:
        result = json.load(response)
    scores = alignment.extract_scores(result, ids, len(ids) - 1)
    path.with_name("teacher-probe.json").write_text(json.dumps({"passed": True, "scored_tokens": len(scores)}))


def main():
    path = Path(sys.argv[1])
    if sys.argv[2] == "--probe":
        probe(path)
        return
    service = json.loads(path.read_text())
    os.execv(sys.executable, command(service, int(sys.argv[2])))


if __name__ == "__main__":
    main()
