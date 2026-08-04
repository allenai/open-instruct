"""A local labelling UI. No install, no dependencies, saves after every answer.

    python -m projects.pedagogy_rm.label_ui \
        --units data/label_slices/slice_1.json --out data/labels/sophia.json

Then open http://localhost:8765.

THE RUBRIC IS ON SCREEN, next to every question, at all times. That is the whole
design. A rubric in a separate file gets read once and then approximated from
memory, and the memory drifts over an afternoon - which is one plausible reason
the previous round's holistic scale collapsed to 39% agreement. Anchors you are
reading as you answer cannot drift.

KEYBOARD. Press 1/2/3 for the highlighted dimension and it advances to the next
one; the sixth answer saves and loads the next turn. Hands never leave the
number row, which matters at 30+ items. ``f`` flags, ``b`` goes back, ``u``
undoes the last answer.

SAVING. Every answer POSTs to the server, which rewrites the output file
immediately. Closing the tab, losing power, or killing the server loses at most
the answer in flight, and reopening resumes where you stopped. Labels are the
expensive part of this project; they are not kept in browser memory.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import threading
import webbrowser

from projects.pedagogy_rm.rubric import DIMENSIONS

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pedagogy labelling</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#272b35; --ink:#e8eaf0; --dim:#9aa3b2;
    --accent:#6ea8fe; --good:#4ade80; --warn:#fbbf24;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:10px 18px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:5; }
  header b { font-size:14px; letter-spacing:.02em; }
  .bar { flex:1; height:6px; background:#262a33; border-radius:3px; overflow:hidden; }
  .bar > i { display:block; height:100%; background:var(--accent); width:0; transition:width .2s; }
  .muted { color:var(--dim); font-size:13px; }
  main { display:grid; grid-template-columns: minmax(0,1fr) 480px; gap:20px; padding:20px; align-items:start; max-width:1500px; margin:0 auto; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  .card + .card { margin-top:14px; }
  .label { font-size:11px; text-transform:uppercase; letter-spacing:.09em; color:var(--dim); margin-bottom:6px; }
  .q { font-size:16px; }
  .choices { margin:10px 0 0; padding-left:20px; color:var(--dim); font-size:14px; }
  /* The answer is shown because two dimensions cannot be judged without it:
     'leak' is about how close the turn comes to it, and 'correct' is about
     whether the turn's claims are true. Guessing the answer while rating those
     would add rater noise that has nothing to do with the turn. */
  .choices li.gold { color:var(--good); font-weight:600; }
  .choices li.gold::after { content:" — correct answer"; font-weight:400; opacity:.75; font-size:12px; }
  .goldline { margin-top:10px; color:var(--good); font-size:14px; }
  .turn { white-space:pre-wrap; }
  .tutor { border-left:3px solid var(--accent); padding-left:14px; font-size:16px; }
  .student { border-left:3px solid #4b5563; padding-left:14px; color:#c7cdd8; }
  .dim { border:1px solid var(--line); border-radius:9px; padding:11px 13px; margin-bottom:10px; background:#141821; }
  .dim.active { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
  .dim.done { border-color:#2f5d3f; }
  .dim h4 { margin:0 0 3px; font-size:14px; }
  .dim h4 span { color:var(--dim); font-weight:400; }
  .dim .qq { color:var(--dim); font-size:13px; margin-bottom:9px; }
  .opt { display:flex; gap:9px; align-items:flex-start; padding:6px 8px; border-radius:6px; cursor:pointer; font-size:13px; color:#cdd3de; }
  .opt:hover { background:#1d2331; }
  .opt kbd { flex:none; background:#232937; border:1px solid #333a4a; border-bottom-width:2px; border-radius:4px; padding:0 6px; font:12px ui-monospace,monospace; color:#aab3c4; }
  .opt.sel { background:#1b3a2a; color:#eafff1; }
  .opt.sel kbd { background:#2f6b47; border-color:#3c8659; color:#fff; }
  footer { padding:0 20px 26px; max-width:1500px; margin:0 auto; color:var(--dim); font-size:13px; }
  button { background:#232937; color:var(--ink); border:1px solid #333a4a; border-radius:6px; padding:6px 11px; cursor:pointer; font-size:13px; }
  button:hover { background:#2b3243; }
  #flagbox { width:100%; margin-top:6px; background:#141821; color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:7px 9px; font:14px inherit; }
  .done-screen { text-align:center; padding:70px 20px; }
</style></head><body>
<header>
  <b>pedagogy labelling</b>
  <div class="bar"><i id="prog"></i></div>
  <span class="muted" id="count"></span>
  <button onclick="back()">&larr; back (b)</button>
  <button onclick="undo()">undo (u)</button>
</header>
<main id="main"></main>
<footer id="foot"></footer>
<script>
let DIMS = [], unit = null, answers = {}, idx = 0, total = 0, active = 0, done = 0;

async function load(direction) {
  const r = await fetch('/api/next' + (direction ? '?dir=' + direction : ''));
  const d = await r.json();
  DIMS = d.dims; total = d.total; done = d.done; idx = d.index;
  unit = d.unit; answers = d.existing || {}; active = 0;
  while (active < DIMS.length && answers[DIMS[active].key] != null) active++;
  if (active >= DIMS.length) active = DIMS.length - 1;
  render();
}

function render() {
  document.getElementById('prog').style.width = (100 * done / total) + '%';
  document.getElementById('count').textContent = done + ' / ' + total + ' labelled';
  if (!unit) {
    document.getElementById('main').innerHTML =
      '<div class="card done-screen"><h2>All done.</h2><p class="muted">' + done +
      ' labelled. The file is written; you can close this tab.</p></div>';
    document.getElementById('foot').textContent = '';
    return;
  }
  const norm = s => String(s == null ? '' : s).trim().replace(/\\s+/g, ' ').toLowerCase();
  const opts = (unit.choices || []).map((c, i) =>
    '<li class="' + (unit.gold != null && norm(c) === norm(unit.gold) ? 'gold' : '') + '">' +
    String.fromCharCode(65 + i) + '. ' + esc(c) + '</li>').join('');
  // Free-response items, or a gold string that matches no option, still need the
  // answer visible - so fall back to stating it rather than showing nothing.
  const goldShown = (unit.choices || []).some(c => unit.gold != null && norm(c) === norm(unit.gold));
  const goldLine = (!goldShown && unit.gold)
    ? '<div class="goldline">correct answer: ' + esc(unit.gold) + '</div>' : '';
  const left =
    '<div><div class="card"><div class="label">Question</div><div class="q">' + esc(unit.question) + '</div>' +
      (opts ? '<ul class="choices">' + opts + '</ul>' : '') + goldLine + '</div>' +
    '<div class="card"><div class="label">Student, just before</div>' +
      '<div class="turn student">' + esc(unit.student_before) + '</div></div>' +
    '<div class="card"><div class="label">Tutor turn &mdash; rate this</div>' +
      '<div class="turn tutor">' + esc(unit.tutor_turn) + '</div></div></div>';

  const right = '<div>' + DIMS.map((d, i) => {
    const sel = answers[d.key];
    const rows = d.anchors.map(a =>
      '<div class="opt' + (sel === a.score ? ' sel' : '') + '" onclick="pick(' + i + ',' + a.score + ')">' +
      '<kbd>' + a.score + '</kbd><div>' + esc(a.text) + '</div></div>').join('');
    return '<div class="dim' + (i === active ? ' active' : '') + (sel != null ? ' done' : '') + '">' +
      '<h4>' + d.key + ' <span>' + d.lo + '&ndash;' + d.hi + '</span></h4>' +
      '<div class="qq">' + esc(d.question) + '</div>' + rows + '</div>';
  }).join('') +
    '<div class="card"><div class="label">Flag (f) &mdash; if the rubric did not fit, say so instead of guessing</div>' +
    '<input id="flagbox" placeholder="e.g. the turn answers a question the student never asked" ' +
    'value="' + esc(answers.flag || '') + '" onchange="saveFlag(this.value)"></div></div>';

  document.getElementById('main').innerHTML = left + right;
  document.getElementById('foot').innerHTML =
    'item ' + (idx + 1) + ' &middot; <span class="muted">' + esc(unit.style || '') + '</span> &middot; ' +
    '<span class="muted">press 1/2/3 &middot; b back &middot; u undo &middot; f flag</span>';
}

function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

async function pick(i, score) {
  const key = DIMS[i].key;
  const first = answers[key] == null;
  answers[key] = score;
  await fetch('/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: unit.id, key: key, value: score }) });
  let next = i + 1;
  while (next < DIMS.length && answers[DIMS[next].key] != null) next++;
  if (next >= DIMS.length) {
    if (first && DIMS.every(d => answers[d.key] != null)) { await load('next'); return; }
    next = DIMS.length - 1;
  }
  active = next; render();
}

async function saveFlag(text) {
  answers.flag = text;
  await fetch('/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: unit.id, key: 'flag', value: text }) });
}

async function undo() {
  if (!unit) return;
  const filled = DIMS.map(d => d.key).filter(k => answers[k] != null);
  if (!filled.length) return back();
  const key = filled[filled.length - 1];
  delete answers[key];
  await fetch('/api/label', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: unit.id, key: key, value: null }) });
  active = DIMS.findIndex(d => d.key === key); render();
}

function back() { load('back'); }

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') { if (e.key === 'Escape') e.target.blur(); return; }
  if (e.key >= '1' && e.key <= '9' && unit) {
    const d = DIMS[active], s = parseInt(e.key, 10);
    if (s >= d.lo && s <= d.hi) pick(active, s);
  } else if (e.key === 'b') back();
  else if (e.key === 'u') undo();
  else if (e.key === 'f') { const b = document.getElementById('flagbox'); if (b) { b.focus(); e.preventDefault(); } }
  else if (e.key === 'ArrowDown') { active = Math.min(active + 1, DIMS.length - 1); render(); }
  else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); render(); }
});
load();
</script></body></html>
"""


class State:
    """Units, labels, and a cursor. Writes the whole file on every change.

    Rewriting 600 small records costs nothing and removes a class of bug that
    matters here more than speed: a partially written label file after a crash.
    """

    def __init__(self, units: list[dict], out: str) -> None:
        self.units = units
        self.out = out
        self.labels: dict[str, dict] = {}
        if os.path.exists(out):
            with open(out) as handle:
                blob = json.load(handle)
            for record in blob.get("labels", []):
                self.labels[record["id"]] = record
            print(f"resuming: {len(self.labels)} already in {out}")
        self.index = self._first_unlabelled()
        self.lock = threading.Lock()

    def _complete(self, record: dict | None) -> bool:
        return bool(record) and all(d.key in record for d in DIMENSIONS)

    def _first_unlabelled(self) -> int:
        for i, unit in enumerate(self.units):
            if not self._complete(self.labels.get(unit["id"])):
                return i
        return len(self.units)

    def save(self) -> None:
        tmp = self.out + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(
                {
                    "schema": "pedagogy-rm/labels-v1",
                    "rater": os.path.basename(self.out).removesuffix(".json"),
                    "labels": list(self.labels.values()),
                },
                handle,
                indent=1,
            )
        os.replace(tmp, self.out)  # atomic, so a crash mid-write cannot truncate the file

    def set(self, unit_id: str, key: str, value) -> None:
        with self.lock:
            record = self.labels.setdefault(unit_id, {"id": unit_id})
            if value is None or value == "":
                record.pop(key, None)
            else:
                record[key] = value
            self.save()

    def done_count(self) -> int:
        return sum(1 for r in self.labels.values() if self._complete(r))


def make_handler(state: State):
    dims_payload = [
        {
            "key": d.key,
            "question": d.question,
            "lo": d.lo,
            "hi": d.hi,
            "anchors": [{"score": s, "text": t} for s, t in sorted(d.anchors.items())],
        }
        for d in DIMENSIONS
    ]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # the terminal is for the labeller, not the server
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/next"):
                if "dir=next" in self.path:
                    state.index += 1
                elif "dir=back" in self.path:
                    state.index = max(0, state.index - 1)
                while (
                    state.index < len(state.units)
                    and "dir=" not in self.path
                    and state._complete(state.labels.get(state.units[state.index]["id"]))
                ):
                    state.index += 1
                unit = state.units[state.index] if state.index < len(state.units) else None
                payload = {
                    "dims": dims_payload,
                    "unit": unit,
                    "existing": state.labels.get(unit["id"], {}) if unit else {},
                    "index": state.index,
                    "total": len(state.units),
                    "done": state.done_count(),
                }
                self._send(200, json.dumps(payload).encode(), "application/json")
            else:
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state.set(body["id"], body["key"], body.get("value"))
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", required=True, help="slice json from build_label_set")
    parser.add_argument("--out", required=True, help="labels json; resumed if it exists")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=0, help="stop after N units, 0 for all")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    with open(args.units) as handle:
        units = json.load(handle)["units"]
    if args.limit:
        units = units[: args.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    state = State(units, args.out)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), make_handler(state)) as server:
        url = f"http://localhost:{args.port}"
        print(f"{len(units)} units, {state.done_count()} already labelled")
        print(f"labelling at {url}   (ctrl-c to stop; progress is saved after every answer)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\nstopped. {state.done_count()} complete in {args.out}")


if __name__ == "__main__":
    main()
