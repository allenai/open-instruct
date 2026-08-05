"""Two tutor turns for the same moment, side by side, one keystroke to choose.

    python projects/pedagogy_rm/prefer_ui.py \
        --pool data/eval/pool.json --pairs data/eval/pairs.json --out data/eval/prefs_sophia.json

A SEPARATE SCRIPT FROM label_ui.py, WHICH IS A JUDGEMENT AND NOT AN OVERSIGHT. The two
tasks look similar and are not: rating asks for five numbers about one turn against an
absolute scale, and this asks for one comparison between two turns and no scale at all.
Sharing a file would mean one HTML template branching on mode in a dozen places, and the
keyboard map - the part that decides whether a hundred judgements take ten minutes or
forty - would have to mean different things in each. Merging them is easy later if the
UIs converge; unpicking a merged one is not.

WHAT THIS BUYS OVER RATING, given that the pool is being rated anyway. Preference is the
lower-variance instrument: there is no scale to hold steady across a session, no drift
between today's 2 and last week's 2, and the comparison is within a moment so question
difficulty cancels. It resolves a small improvement that absolute ratings would leave
inside the noise. What it cannot do is say which dimension moved, or catch the probe
scoring a turn higher than a person would - both need the absolute labels. Run both.

TIES ARE A FIRST-CLASS ANSWER. Forcing a choice between two turns that are genuinely alike
manufactures a signal out of a coin flip, and with three arms most pairs at an early
training step really are alike. `=` records that, and the analysis counts it rather than
dropping it.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import threading
import webbrowser

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>preference</title>
<style>
 body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; background: #f6f6f7; color: #111; }
 header { padding: 10px 16px; background: #fff; border-bottom: 1px solid #ddd; display: flex; gap: 16px; }
 header b { font-weight: 600; }
 .context { margin: 16px; padding: 12px 16px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }
 .context .q { font-weight: 600; }
 .context .s { margin-top: 8px; color: #444; }
 .pair { display: flex; gap: 16px; margin: 0 16px 16px; }
 .side { flex: 1; background: #fff; border: 2px solid #ddd; border-radius: 6px; padding: 14px 16px; }
 .side.pick { border-color: #1a7f37; background: #f2fbf4; }
 .side h3 { margin: 0 0 8px; font-size: 13px; letter-spacing: .06em; color: #666; text-transform: uppercase; }
 .keys { margin: 0 16px 24px; color: #555; }
 kbd { background: #eee; border: 1px solid #ccc; border-radius: 4px; padding: 1px 6px; font-family: inherit; }
 .done { color: #1a7f37; font-weight: 600; }
</style></head><body>
<header><b id="pos"></b><span id="done" class="done"></span><span id="mid"></span></header>
<div class="context"><div class="q" id="q"></div><div class="s" id="s"></div></div>
<div class="pair">
  <div class="side" id="lwrap"><h3>Left &mdash; press J</h3><div id="left"></div></div>
  <div class="side" id="rwrap"><h3>Right &mdash; press K</h3><div id="right"></div></div>
</div>
<div class="keys"><kbd>J</kbd> left is better &nbsp; <kbd>K</kbd> right is better &nbsp;
 <kbd>=</kbd> too close to call &nbsp; <kbd>&larr;</kbd> back &nbsp; <kbd>&rarr;</kbd> skip</div>
<script>
let cur = null;
async function load(dir) {
  const r = await fetch('/api/next' + (dir ? '?dir=' + dir : ''));
  cur = await r.json();
  if (cur.finished) {
    document.getElementById('pos').textContent = 'all done';
    document.getElementById('q').textContent = 'Every pair has an answer. Close the tab.';
    document.getElementById('s').textContent = '';
    document.getElementById('left').textContent = '';
    document.getElementById('right').textContent = '';
    return;
  }
  document.getElementById('pos').textContent = (cur.index + 1) + ' / ' + cur.total;
  document.getElementById('done').textContent = cur.done + ' answered';
  document.getElementById('q').textContent = cur.question;
  document.getElementById('s').textContent = 'Student: ' + cur.student_before;
  document.getElementById('left').textContent = cur.left;
  document.getElementById('right').textContent = cur.right;
  document.getElementById('lwrap').className = 'side' + (cur.choice === 'left' ? ' pick' : '');
  document.getElementById('rwrap').className = 'side' + (cur.choice === 'right' ? ' pick' : '');
  document.getElementById('mid').textContent = cur.choice === 'tie' ? 'recorded: too close' : '';
}
async function choose(choice) {
  if (!cur || cur.finished) return;
  await fetch('/api/choose', { method: 'POST',
    body: JSON.stringify({ id: cur.id, choice: choice }) });
  load('next');
}
document.addEventListener('keydown', e => {
  if (e.key === 'j' || e.key === 'J') choose('left');
  else if (e.key === 'k' || e.key === 'K') choose('right');
  else if (e.key === '=') choose('tie');
  else if (e.key === 'ArrowLeft') load('back');
  else if (e.key === 'ArrowRight') load('next');
});
load();
</script></body></html>
"""


class State:
    """Pairs, choices, and a cursor. Rewrites the whole file on every answer.

    The same trade label_ui.py makes and for the same reason: a few hundred small records
    cost nothing to rewrite, and an atomic replace removes the possibility of a half-written
    answer file after a crash - which is the failure that would cost a labelling session.
    """

    def __init__(self, pool: list[dict], pairs: list[dict], out: str) -> None:
        self.by_id = {unit["id"]: unit for unit in pool}
        self.pairs = [p for p in pairs if p["left"] in self.by_id and p["right"] in self.by_id]
        if len(self.pairs) != len(pairs):
            print(f"warning: {len(pairs) - len(self.pairs)} pairs name turns the pool does not hold")
        self.out = out
        self.choices: dict[str, dict] = {}
        if os.path.exists(out):
            with open(out) as handle:
                for record in json.load(handle).get("preferences", []):
                    self.choices[record["id"]] = record
            print(f"resuming: {len(self.choices)} already in {out}")
        self.index = self._first_unanswered()
        self.lock = threading.Lock()

    def _first_unanswered(self) -> int:
        for i, pair in enumerate(self.pairs):
            if pair["id"] not in self.choices:
                return i
        return len(self.pairs)

    def save(self) -> None:
        tmp = self.out + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(
                {
                    "schema": "pedagogy-rm/preferences-v1",
                    "rater": os.path.basename(self.out).removesuffix(".json"),
                    "preferences": list(self.choices.values()),
                },
                handle,
                indent=1,
            )
        os.replace(tmp, self.out)

    def choose(self, pair_id: str, choice: str) -> None:
        if choice not in ("left", "right", "tie"):
            return
        with self.lock:
            pair = next((p for p in self.pairs if p["id"] == pair_id), None)
            if pair is None:
                return
            # The chosen turn's blinded id is recorded, not "left" alone: the sides were
            # drawn at random per pair, so "left" means nothing once this file is joined
            # against the key.
            winner = None if choice == "tie" else pair[choice]
            self.choices[pair_id] = {
                "id": pair_id,
                "moment": pair["moment"],
                "left": pair["left"],
                "right": pair["right"],
                "choice": choice,
                "winner": winner,
            }
            self.save()

    def payload(self) -> dict:
        if self.index >= len(self.pairs):
            return {"finished": True, "done": len(self.choices)}
        pair = self.pairs[self.index]
        left, right = self.by_id[pair["left"]], self.by_id[pair["right"]]
        answered = self.choices.get(pair["id"])
        return {
            "finished": False,
            "id": pair["id"],
            "index": self.index,
            "total": len(self.pairs),
            "done": len(self.choices),
            "question": left.get("question", ""),
            "student_before": left.get("student_before", ""),
            "left": left.get("tutor_turn", ""),
            "right": right.get("tutor_turn", ""),
            "choice": answered["choice"] if answered else None,
        }


def make_handler(state: State):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            return

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/next"):
                if "dir=next" in self.path:
                    state.index = min(state.index + 1, len(state.pairs))
                elif "dir=back" in self.path:
                    state.index = max(state.index - 1, 0)
                self._send(200, json.dumps(state.payload()).encode(), "application/json")
            else:
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state.choose(html.unescape(str(body.get("id", ""))), str(body.get("choice", "")))
            self._send(200, b"{}", "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/eval/pool.json", help="blinded turns from build_blind_set")
    parser.add_argument("--pairs", default="data/eval/pairs.json")
    parser.add_argument("--out", required=True, help="preferences json; resumed if it exists")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    with open(args.pool) as handle:
        pool = json.load(handle)["units"]
    with open(args.pairs) as handle:
        pairs = json.load(handle)["pairs"]

    state = State(pool, pairs, args.out)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(state.pairs)} pairs, {len(state.choices)} answered -> {url}")
    print("  J left, K right, = too close, arrows to move")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped with {len(state.choices)} of {len(state.pairs)} answered; rerun to resume")


if __name__ == "__main__":
    main()
