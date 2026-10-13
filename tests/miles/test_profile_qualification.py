"""Check workload coverage and interpretation of qualification measurements."""

import json
import threading
from http import server
from types import SimpleNamespace
from urllib import error

import pytest

qualification = pytest.importorskip("open_instruct.miles.distillation.profile_qualification")


@pytest.mark.parametrize(
    "status,body",
    [(200, b"Cache flushed.\nPlease check backend logs."), (400, b"Flush cache failed."), (200, b"Unexpected reply")],
)
def test_flush_cache_handles_real_http_text_and_rejects_failures(status, body):
    calls = []

    class Handler(server.BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{http.server_port}"
            if status == 400:
                with pytest.raises(error.HTTPError):
                    qualification.flush_cache(endpoint)
            elif not body.startswith(b"Cache flushed."):
                with pytest.raises(RuntimeError, match="Unexpected cache-flush response"):
                    qualification.flush_cache(endpoint)
            else:
                qualification.flush_cache(endpoint)
            assert calls == ["/flush_cache?timeout=30"]
        finally:
            http.shutdown()
            thread.join()


def test_workloads_cover_real_references_and_context_boundaries():
    items = [
        dict(
            sample_id=str(size),
            rendered=dict(
                input="bare",
                metadata={
                    "opd_teacher_prompt_tokens": size,
                    "opd_teacher_prompt": "f" * size,
                    "opd_answer_prompt": "answer",
                },
            ),
        )
        for size in (4, 16, 8)
    ]
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [ord(c) for c in text])
    cases, selected = qualification.workloads(items, tokenizer, dict(prefix=[1, 2], tokens=[3, 4, 5]))
    assert [x["sample_id"] for x in selected] == ["4", "8", "16"]
    assert all(len(c["tokens"]) == 16384 for c in cases if c["kind"] != "boundary")
    lengths = {len(c["prefix"]) + len(c["tokens"]) for c in cases if c["kind"] == "boundary"}
    assert {8191, 8192, 8193, 16383, 16384, 16385, 32760, 65528} <= lengths
    assert max(len(c["prefix"]) + len(c["tokens"]) for c in cases) == 65528


def test_roles_preserve_production_cache_graph_and_context_settings():
    student = qualification.command("/model", 123, "student")
    teacher = qualification.command("/model", 124, "teacher")
    assert student[student.index("--context-length") + 1] == "32768"
    assert teacher[teacher.index("--context-length") + 1] == "65536"
    assert "--disable-radix-cache" not in student and "--disable-cuda-graph" not in student
    assert "--disable-radix-cache" in teacher and "--disable-cuda-graph" in teacher
    assert student[student.index("--cuda-graph-max-bs-decode") + 1] == "8"


def test_summary_excludes_first_request_timings_and_reports_nonexact(tmp_path):
    (tmp_path / "comparisons.jsonl").write_text(json.dumps(dict(mode="strict", mae=0.01)) + "\n")
    values = [
        dict(
            mode=mode,
            role="student",
            case="mixed",
            batch=4,
            repeat=repeat,
            input_tokens=1000,
            seconds=seconds,
            phase="warm" if repeat else "first_request",
        )
        for mode, repeat, seconds in (("baseline", 0, 100), ("baseline", 1, 2), ("strict", 0, 200), ("strict", 1, 1))
    ]
    (tmp_path / "timings.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    result = qualification.summarize(tmp_path)
    assert not result["all_exact"]
    assert result["throughput"][0]["strict_over_baseline"] == 2
