import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.miles import launch_readiness_cpu, prepare_stdio_exercise, readiness_services

from open_instruct.miles import code_rewards


def test_real_http_retry_budget_and_recovery(monkeypatch):
    class Healthy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["tests"]
            body = b'{"results":[1]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Healthy)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(code_rewards, "RETRY", code_rewards.RETRY.new(backoff_factor=0))
    monkeypatch.setattr(code_rewards, "_SESSION", None)
    try:
        report = readiness_services.exercise(f"http://127.0.0.1:{server.server_port}")
        assert report["passed"] and len(report["cases"]) == 6
        exhausted = report["cases"][2]
        assert exhausted["forwarded"] == 0
        assert len(exhausted["attempts"]) == code_rewards.RETRY.total + 1
        assert report["cases"][3]["score"] == 1
    finally:
        if code_rewards._SESSION is not None:
            code_rewards._SESSION.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_service_launcher_preserves_url_and_has_no_mode_argument():
    url = "https://example.invalid/prod"
    spec = launch_readiness_cpu.specification("image", "services", [url], b"pass")
    assert spec["tasks"][0]["arguments"][0].endswith("python /output/readiness_cpu.py " + url)
    assert spec["tasks"][0]["resources"]["gpuCount"] == 0


def test_stdio_selection_rejects_example_only_prompts():
    wrapper = "where CODE is the solution for the problem.\n\n"
    suffix = "\nWrite Python code to solve the problem."
    assert not prepare_stdio_exercise.has_statement(wrapper + "Example\nInput\n5\nOutput\n5" + suffix)
    assert not prepare_stdio_exercise.has_statement(
        wrapper + "Return the Nth Even Number\nThe input will not be 0." + suffix
    )

    assert not prepare_stdio_exercise.has_statement(
        wrapper + "Time Limit: 8 sec / Memory Limit: 64 MB\nExample\nInput\n5\nOutput\n5" + suffix
    )
    assert prepare_stdio_exercise.has_statement(
        wrapper + "Calculate the sum of two given integers.\nInput\nTwo integers a and b.\nOutput\nTheir sum." + suffix
    )
