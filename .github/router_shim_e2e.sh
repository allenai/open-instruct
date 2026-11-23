#!/bin/bash
# TEMP (never merged): run inside the built MILES image. Real sgl-router against a fake engine.
set -uo pipefail
cd /tmp
python -c "import sys, sglang_router; print(sys.version); print('sglang_router', sglang_router.__version__)"
ls -l /opt/core-rl/lib/no_tcp_user_timeout.so
SITE=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))'); cat "$SITE/oi_router_preload.pth"

cat > fake_engine.py <<'EOF'
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self._send({"model_path": "fake", "is_generation": True})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        time.sleep(2)
        self._send({"text": "ok", "meta_info": {"id": "x", "finish_reason": {"type": "stop"}}})
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", 31000), H).serve_forever()
EOF
python fake_engine.py & ENGINE=$!
sleep 2

run_router() {  # $1 = label, rest = env assignments
  local label=$1; shift
  env "$@" python -m sglang_router.launch_router --host 127.0.0.1 --port 30000 \
      --worker-urls http://127.0.0.1:31000 > "router_$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 60); do curl -sf -o /dev/null http://127.0.0.1:30000/health && break; sleep 1; done
  local code
  code=$(curl -s -o "gen_$label.json" -w '%{http_code}' -X POST http://127.0.0.1:30000/generate \
      -H 'Content-Type: application/json' -d '{"text": "hi", "sampling_params": {"max_new_tokens": 1}}')
  echo "== $label: pid=$pid /generate HTTP $code body=$(cat "gen_$label.json")"
  echo "   cmdline: $(tr '\0' ' ' < /proc/$pid/cmdline)"
  echo "   LD_PRELOAD: $(tr '\0' '\n' < /proc/$pid/environ | grep '^LD_PRELOAD=' || echo '(unset)')"
  if grep -q no_tcp_user_timeout.so /proc/$pid/maps; then echo "   shim mapped: yes"; else echo "   shim mapped: no"; fi
  grep -n "no_tcp_user_timeout" "router_$label.log" || echo "   (no shim log line)"
  kill $pid; wait $pid 2>/dev/null
  sleep 1
}
run_router default
run_router optout OI_MILES_ROUTER_SHIM=0

# Any other Python process (engines, trainer, driver) must not load the shim.
OTHER=$(python -c "print('no_tcp_user_timeout' in open('/proc/self/maps').read())"); echo "other python maps shim: $OTHER"
kill $ENGINE

fail=0
grep -q "dropped TCP_USER_TIMEOUT" router_default.log || { echo "FAIL: default router did not log the dropped option"; fail=1; }
grep -q "no_tcp_user_timeout" router_optout.log && { echo "FAIL: opt-out router loaded the shim"; fail=1; }
[ "$OTHER" = False ] || { echo "FAIL: a non-router Python loaded the shim"; fail=1; }
echo "--- router_default.log (tail)"; tail -20 router_default.log
exit $fail
