"""Prepare immutable teacher weights and exec a supervised SGLang scoring server."""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from open_instruct.miles import opd_alignment


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


def probe(path):
    service = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(service["snapshot"], trust_remote_code=True)
    context = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1 + 1?"}],
        tokenize=False,
        add_generation_prompt=True,
        **service["chat_template_kwargs"],
    )
    ids = tokenizer.encode(context + "2", add_special_tokens=False)
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    request = urllib.request.Request(
        service["endpoint"] + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=service["request_timeout"]) as response:
        result = json.load(response)
    scores, _ = opd_alignment.extract_scores(result, ids, list(range(1, len(ids))))
    path.with_name("teacher-probe.json").write_text(json.dumps({"passed": True, "scored_tokens": len(scores)}))


def main():
    path = Path(sys.argv[1])
    if sys.argv[2] == "--probe":
        probe(path)
        return
    port = int(sys.argv[2])
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
