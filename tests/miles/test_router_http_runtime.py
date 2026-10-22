"""Exercise the real router with an HTTP worker and an SFT-only template tag."""

import json
import socket
import subprocess
import sys
import threading
import time
from http import server

import requests
from tokenizers import Tokenizer, models


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait(check, process, log):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert process.poll() is None, log.read_text()
        try:
            value = check()
            if value:
                return value
        except requests.ConnectionError:
            pass
        time.sleep(0.1)
    raise AssertionError(log.read_text())


def test_http_worker_skips_implicit_tokenizer_and_forwards_tokens(tmp_path):
    # Valid for Transformers' assistant-span tracking, unsupported by MiniJinja.
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(str(tokenizer / "tokenizer.json"))
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": "{% generation %}{{ messages[0]['content'] }}{% endgeneration %}",
            }
        )
    )
    received = []

    class Worker(server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                {
                    "model_path": str(tokenizer),
                    "tokenizer_path": str(tokenizer),
                    "served_model_name": "template-probe",
                    "dp_size": 1,
                    "tp_size": 1,
                    "is_generation": True,
                }
            )

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.reply({"text": "hello", "meta_info": {"finish_reason": {"type": "stop"}}})

    worker = server.ThreadingHTTPServer(("127.0.0.1", 0), Worker)
    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()
    router_url = f"http://127.0.0.1:{_port()}"
    log = tmp_path / "router.log"
    process = None
    try:
        with log.open("w") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    router_url.rsplit(":", 1)[1],
                    "--prometheus-port",
                    str(_port()),
                    "--log-level",
                    "info",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            _wait(lambda: requests.get(f"{router_url}/workers", timeout=2).status_code == 200, process, log)
            response = requests.post(
                f"{router_url}/workers",
                json={"url": f"http://127.0.0.1:{worker.server_port}", "worker_type": "regular"},
                timeout=5,
            )
            response.raise_for_status()
            _wait(lambda: "Skipping automatic tokenizer registration for HTTP model" in log.read_text(), process, log)
            workers = _wait(
                lambda: requests.get(f"{router_url}/workers", timeout=2).json().get("workers"), process, log
            )
            assert requests.get(f"{router_url}/v1/tokenizers", timeout=2).json()["tokenizers"] == []
            payload = {"input_ids": [1], "sampling_params": {"max_new_tokens": 1}}
            response = requests.post(f"{router_url}/generate", json=payload, timeout=5)
            response.raise_for_status()
            assert response.json()["text"] == "hello"
            assert received[-1]["input_ids"] == payload["input_ids"]
            assert received[-1]["sampling_params"]["max_new_tokens"] == 1
            response = requests.delete(f"{router_url}/workers/{workers[0]['id']}", timeout=5)
            response.raise_for_status()
            _wait(lambda: not requests.get(f"{router_url}/workers", timeout=2).json()["workers"], process, log)
            assert "Failed to load tokenizer" not in log.read_text()
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        worker.shutdown()
        worker.server_close()
        thread.join(timeout=5)
