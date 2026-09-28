import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.miles import readiness_services

from open_instruct.miles.rewards import code_rewards


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
    try:
        report = readiness_services.exercise(f"http://127.0.0.1:{server.server_port}")
        assert report["passed"] and len(report["cases"]) == 6
        exhausted = report["cases"][2]
        assert exhausted["forwarded"] == 0
        assert len(exhausted["attempts"]) == code_rewards.RETRY.total + 1
        assert report["cases"][3]["score"] == 1
    finally:
        code_rewards._get_session().close()
        code_rewards.service.http_session.cache_clear()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
