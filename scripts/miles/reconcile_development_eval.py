"""Automatically correct saved development evaluations without restarting training.

Runs in the pinned CPU evaluator image. Original generations and metrics remain
immutable; corrected metrics use the distinct development_gold_v2 task name.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import development_eval


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Publisher:
    """Scoring and W&B live in different Python environments in evaluator images."""

    def __init__(self, script, python):
        self.script = script
        self.python = python

    def publish(self, receipt, output):
        path = output / "publication-receipt.json"
        atomic_json(path, receipt)
        subprocess.run(
            [self.python, str(self.script), "publish", str(path), "--results", str(output)], check=True, timeout=300
        )


def read_development(source):
    panel = [json.loads(line) for line in (source / "selected-panel.jsonl").open()]
    selected = {item["id"]: item for item in panel if item["task"] == "development"}
    if not selected:
        return None
    if any(item["samples"] != 4 for item in selected.values()):
        raise ValueError("This correction protocol requires four samples per question")
    for item in selected.values():
        development_eval.gold_answers(item["label"])
    expected = {(qid, r) for qid in selected for r in range(4)}
    rows = {}
    for line in (source / "generations.jsonl").open():
        row = json.loads(line)
        if row["task"] != "development":
            continue
        identity = row["id"], row["replicate"]
        if identity in rows or identity not in expected:
            raise ValueError(f"Duplicate or unexpected development response: {identity}")
        rows[identity] = row
    if set(rows) != expected:
        return None  # Generation is still in progress. Never score partial coverage.
    return selected, rows


def reconcile(source, destination, receipt, publisher):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        return False
    if not (destination / "metrics.json").exists():
        data = read_development(source)
        if data is None:
            return False
        panel, rows = data
        scored = []
        source_hash = hashlib.sha256()
        temporary = destination / "responses.tmp"
        with temporary.open("w") as stream:
            for identity in sorted(rows):
                row = rows[identity]
                source_hash.update(json.dumps(row, sort_keys=True).encode())
                result = development_eval.score_one(row, panel[row["id"]])
                result["text_sha256"] = hashlib.sha256(result.pop("text").encode()).hexdigest()
                stream.write(json.dumps(result) + "\n")
                scored.append(result)
        temporary.replace(destination / "responses.jsonl")
        metrics = {
            key: {"mean": sum(float(row[key]) for row in scored) / len(scored)}
            for key in ("score", "completed_final_score", "capped", "tokens")
        }
        correct = {qid: 0 for qid in panel}
        for row in scored:
            correct[row["id"]] += int(row["completed_final_score"])
        metrics.update(
            pass_at_1={"mean": sum(correct.values()) / len(scored)},
            pass_at_2={"mean": sum(1 - (4 - c) * (3 - c) / 12 for c in correct.values()) / len(panel)},
            pass_at_4={"mean": sum(c > 0 for c in correct.values()) / len(panel)},
            all_four_correct={"mean": sum(c == 4 for c in correct.values()) / len(panel)},
        )
        atomic_json(
            destination / "provenance.json",
            {
                "source": str(source),
                "generation_sha256": source_hash.hexdigest(),
                "panel_sha256": hashlib.sha256((source / "selected-panel.jsonl").read_bytes()).hexdigest(),
                "scorer_sha256": hashlib.sha256(Path(development_eval.__file__).read_bytes()).hexdigest(),
                "questions": len(panel),
                "responses": len(scored),
                "correction": "Flat accepted-gold list; original generations, extraction, and completion gates",
            },
        )
        atomic_json(
            destination / "metrics.json",
            {"tasks": [dict(task="development_gold_v2", instances_saved=len(scored), metrics=metrics)]},
        )
    update = int(source.name.split("-")[1])
    publisher.publish(dict(receipt, update=update, group_id="development-gold-v2-" + source.name), destination)
    atomic_json(destination / "complete.json", {"update": update, "status": "complete"})
    print(json.dumps({"corrected_update": update, "output": str(destination)}), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--publisher-python", default="python", help="Image Python containing W&B")
    parser.add_argument("--watch-hours", type=float, default=96)
    args = parser.parse_args()
    receipts = [json.loads(path.read_text()) for path in args.receipt]
    publisher = Publisher(args.publisher, args.publisher_python)
    deadline = time.monotonic() + args.watch_hours * 3600
    while True:
        for receipt in receipts:
            root = Path(receipt["evaluation"]["root"]) / "evaluation"
            for source in sorted((root / "results").glob("update-*")):
                if not (source / "selected-panel.jsonl").exists() or not (source / "generations.jsonl").exists():
                    continue
                destination = root / "corrections" / "development-gold-v2" / source.name
                try:
                    # Resume may create a new W&B run; use this evaluation's own receipt.
                    current = json.loads((root / (source.name + ".json")).read_text())
                    if current["runner_sha256"] != receipt["runner_sha256"]:
                        raise ValueError("Evaluation publisher revision changed; review before correcting")
                    reconcile(source, destination, current, publisher)
                except Exception as error:
                    destination.mkdir(parents=True, exist_ok=True)
                    print(json.dumps({"warning": str(error), "source": str(source), "retry_seconds": 60}), flush=True)
                    atomic_json(destination / "last-error.json", {"error": repr(error), "time": time.time()})
        if time.monotonic() >= deadline:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
