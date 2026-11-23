"""Full Minerva MATH + AIME 2026 eval sweep over every saved checkpoint of the 2.7B 8k and 32k runs.

For each checkpoint: export it to HF with an empty-loop MILES run (`miles.load` = its preserved copy, num_rollouts = its
update), run the extra-eval job on the export (tools/format_probe.py, as for the 8k final / 32k u250 evals), and score it
with tools/score_extra_evals.py. Checkpoints that already have an HF export skip the export; the 32k final waits for that
run to finish and export itself. At most --max-jobs Beaker jobs from this script run at once. Ends with one CSV table.

Usage (from anywhere): python eval_sweep_orchestrator.py [--max-jobs 2] [--dry-run]
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

O = Path("/weka/oe-adapt-default/jacobm/olmicro")
OI = O / "open-instruct"
SP = Path("/tmp/claude-0/-weka-oe-adapt-default-jacobm-olmicro-miles-olmo-core--claude-worktrees-bridge-cse-01TgX3YnCiqBAC4S2yxerMdT/a9337bf4-c3a0-51fb-a081-e1d79b9b351f/scratchpad")
IMAGE = "01M6HMYBFW8D9ZK4SKGJ1XD6HW"
RUN8 = "olmicro-2.7b-lc-math-rl-v2-refresh-8k-20261118"
RUN32_OLD = "olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118"
RUN32 = "olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118"
RUN14 = "olmicro-1.4b-lc-math-rl-refresh-32k-20261118"
# Export templates already validated tonight: (template TOML, load path in it, num_rollouts in it).
TEMPLATES = {
    "8k": (f"{RUN8.replace('-20261118', '')}-u250-export-20261118", f"{O}/preserved-checkpoints/{RUN8}/u0250", 250),
    "32k": ("olmicro-2.7b-lc-math-rl-v2-refresh-32k-u250-export-20261118", f"{O}/preserved-checkpoints/{RUN32}/u0250", 250),
    "32k-old": ("olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-export-20261118", f"{O}/runs/{RUN32_OLD}/checkpoints", 100),
    "1.4b": ("olmicro-1.4b-lc-math-rl-refresh-32k-u250-export-20261118", f"{O}/preserved-checkpoints/{RUN14}/u0250", 250),
}


def checkpoints(which="2.7b"):
    items = []
    if which == "1.4b":
        return [{"tag": f"1.4b-u{u}", "template": "1.4b", "update": u, "load": f"{O}/preserved-checkpoints/{RUN14}/u{u:04d}"} for u in (50, 100, 150, 200, 250, 300, 350)]
    for u in (100, 150, 200, 300, 350):
        items.append({"tag": f"8k-u{u}", "template": "8k", "update": u, "load": f"{O}/preserved-checkpoints/{RUN8}/u{u:04d}"})
    items.append({"tag": "32k-u50", "template": "32k-old", "update": 50, "load": f"{O}/preserved-checkpoints/{RUN32_OLD}/u0050"})
    items.append({"tag": "32k-u100", "hf": f"{O}/runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-export-20261118/export-hf"})
    for u in (150, 200, 300, 350):
        items.append({"tag": f"32k-u{u}", "template": "32k", "update": u, "load": f"{O}/preserved-checkpoints/{RUN32}/u{u:04d}"})
    items.append({"tag": "32k-u400", "final": f"{O}/runs/{RUN32}"})
    return items


def log(*parts):
    print(time.strftime("%H:%M:%SZ", time.gmtime()), *parts, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def job_state(experiment):
    out = sh(["timeout", "60", "beaker", "experiment", "get", experiment, "--format", "json"]).stdout
    try:
        e = json.loads(out)
        e = e[0] if isinstance(e, list) else e
        s = e["jobs"][-1]["status"] if e.get("jobs") else {}
    except (json.JSONDecodeError, KeyError, IndexError):
        return "unknown", None
    if s.get("exited") or s.get("canceled"):
        return "ended", s.get("exitCode")
    return ("running" if s.get("started") else "queued"), None


def make_export(item, dry):
    template, load, rollouts = TEMPLATES[item["template"]]
    name = template.replace("u250", f"u{item['update']}").replace("u100", f"u{item['update']}")
    text = (OI / "runs" / f"{template}.toml").read_text().replace(template, name)
    assert text.count(f'load = "{load}"') == 1 and text.count(f"num_rollouts = {rollouts}\n") == 1, template
    text = text.replace(f'load = "{load}"', f'load = "{item["load"]}"').replace(f"num_rollouts = {rollouts}\n", f"num_rollouts = {item['update']}\n")
    path = OI / "runs" / f"{name}.toml"
    if not path.exists():
        path.write_text(text)
    item["hf"] = f"{O}/runs/{name}/export-hf"
    check = sh(["timeout", "300", "python3", "-m", "open_instruct.miles", "validate", str(path.relative_to(OI))], cwd=OI)
    if check.returncode:
        raise RuntimeError(f"validate failed for {name}: {check.stdout[-400:]} {check.stderr[-400:]}")
    if dry:
        log("dry-run export ok", name)
        return None
    if Path(item["hf"], "config.json").is_file():
        return "done"
    out = sh(["timeout", "600", "python3", "../tools/submit_excluding.py", IMAGE, str(path.relative_to(OI)), "--exclude", "holmes-cs-aus-511.reviz.ai2.in"],
             cwd=OI, env={**__import__("os").environ, "PYTHONPATH": str(OI)})
    url = re.search(r"https://beaker.org/ex/(\w+)", out.stdout)
    if not url:
        raise RuntimeError(f"submit failed for {name}: {out.stdout[-400:]} {out.stderr[-400:]}")
    log("export submitted", item["tag"], url.group(0))
    return url.group(1)


def make_eval(item, dry):
    spec = json.loads((O / "tools/extra_evals_8k-u250_beaker.json").read_text())
    task = spec["tasks"][0]
    old_model = f"{O}/runs/{RUN8.replace('-20261118', '')}-u250-export-20261118/export-hf"
    old_out = f"{O}/monitor/extra-evals-8k-u250-20261118"
    out_dir = f"{O}/monitor/extra-evals-{item['tag']}-20261118"
    arg = task["arguments"][0]
    assert old_model in arg and old_out in arg
    task["arguments"] = [arg.replace(old_model, item["hf"]).replace(old_out, out_dir)]
    task["name"] = f"olmicro-extra-evals-{item['tag']}"
    spec["description"] = spec["description"].replace("8k run update 250", item["tag"])
    path = O / f"tools/extra_evals_{item['tag']}_beaker.json"
    path.write_text(json.dumps(spec, indent=1))
    item["out"] = out_dir
    if dry:
        log("dry-run eval spec", path)
        return None
    if Path(out_dir, "summary.json").is_file():
        return "done"
    out = sh(["timeout", "120", "beaker", "experiment", "create", str(path), "--workspace", "ai2/olmo3p5-training", "--format", "json"])
    e = json.loads(out.stdout)
    e = e[0] if isinstance(e, list) else e
    log("eval submitted", item["tag"], f"https://beaker.org/ex/{e['id']}")
    return e["id"]


def score(item):
    out = Path(item["out"])
    if len(list(out.glob("*.raw.json"))) != 8:
        log("eval incomplete", item["tag"], len(list(out.glob("*.raw.json"))), "of 8 shards")
        return False
    sh(["timeout", "3600", str(SP / "hfenv/bin/python"), str(O / "tools/score_extra_evals.py"), str(out), str(O / "data/olmicro-extra-evals-r1-v1"),
        str(SP / "repos/olmo-eval/src/olmo_eval/evals/extract/math.py"), "--pool", "32"], cwd=SP, env={**__import__("os").environ, "PYTHONPATH": str(OI)})
    s = json.loads((out / "summary.json").read_text())
    log("RESULT", item["tag"], "minerva macro", s["minerva_math"]["macro_oe"], "| aime pass@1", s["aime2026"]["pass@1_oe"], "pass@32", s["aime2026"]["pass@32_oe"])
    return True


def table(which="2.7b"):
    rows = ["checkpoint,minerva_macro_oe,minerva_micro_oe,minerva_macro_mv,aime_pass1_oe,aime_pass4_oe,aime_pass32_oe,aime_pass1_oe_16k,minerva_mean_tokens,aime_mean_tokens"]
    tags = ["base-2.7b"] + [f"8k-u{u}" for u in (100, 150, 200, 250, 300, 350)] + ["8k-final"] + [f"32k-u{u}" for u in (50, 100, 150, 200, 250, 300, 350, 400)]
    if which == "1.4b":
        tags = ["base-1.4b"] + [f"1.4b-u{u}" for u in (50, 100, 150, 200, 250, 300, 350)] + ["1.4b-final"]
    for tag in tags:
        p = O / f"monitor/extra-evals-{tag}-20261118/summary.json"
        if p.is_file():
            s = json.loads(p.read_text()); m, a = s["minerva_math"], s["aime2026"]
            rows.append(",".join(map(str, [tag, m["macro_oe"], m["acc_oe"], m["macro_mv"], a["pass@1_oe"], a["pass@4_oe"], a["pass@32_oe"], a["pass@1_oe_16k"], m["mean_tokens"], a["mean_tokens"]])))
    (O / f"runs/plots/extra-evals-{which}-20261118.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--set", dest="which", choices=("2.7b", "1.4b"), default="2.7b")
    args = parser.parse_args()
    items = checkpoints(args.which)
    if args.dry_run:
        for item in items:
            if "template" in item:
                make_export(item, True)
                make_eval(item, True)
        return
    stop = time.time() + args.hours * 3600
    for item in items:
        item["stage"] = "export" if "template" in item else ("wait_final" if "final" in item else "eval")
    while time.time() < stop and any(i["stage"] != "done" for i in items):
        active = sum(1 for i in items if i.get("job"))
        for item in items:
            if item.get("job"):
                state, code = job_state(item["job"])
                if state != "ended":
                    continue
                item["job"] = None
                if item["stage"] == "export":
                    ok = code == 0 and Path(item["hf"], "config.json").is_file()
                    log("export ended", item["tag"], code, "ok" if ok else "FAILED")
                    item["stage"] = "eval" if ok else "done"
                else:
                    score(item)
                    item["stage"] = "done"
                active -= 1
        for item in items:
            if item["stage"] == "wait_final":
                root = Path(item["final"])
                status = json.loads((root / "workflow.json").read_text()).get("status")
                if status == "complete" and (root / "export-hf/.complete").is_file():
                    item["hf"] = str(root / "export-hf"); item["stage"] = "eval"; log("32k run complete; final export ready")
                elif status == "failed":
                    log("32k run FAILED; skipping its final eval"); item["stage"] = "done"
        for item in items:
            if active >= args.max_jobs:
                break
            if item.get("job") or item["stage"] not in ("export", "eval"):
                continue
            try:
                job = make_export(item, False) if item["stage"] == "export" else make_eval(item, False)
            except Exception as error:  # noqa: BLE001 - log and move on; one bad checkpoint must not stop the sweep
                log("ERROR", item["tag"], error)
                item["stage"] = "done"
                continue
            if job == "done":
                if item["stage"] == "export":
                    item["stage"] = "eval"
                else:
                    score(item); item["stage"] = "done"
                continue
            item["job"] = job
            active += 1
        time.sleep(60)
    table(args.which)


if __name__ == "__main__":
    main()
