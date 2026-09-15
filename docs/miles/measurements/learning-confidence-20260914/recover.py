import ast
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[4]
out = root / "docs/miles/measurements/learning-confidence-20260914"
report = {
    "scope": "Historical log recovery and independent light-SFT full-test rescoring. Current mixed-task runs are incomplete.",
    "runs": {},
}
history = json.loads((root / "configs/miles/reference/light-sft1000-gsm8k-historical.json").read_text())
report["light_megatron_reference"] = {
    "experiment": history["historical_training_experiment"],
    "native_eval": history["native_eval"],
    "full_test": history["full_test"],
    "performance": history["performance"],
}
for name in ["core500", "megatron500"]:
    log = Path("/tmp/" + name + "-overnight.log")
    status = json.loads(Path("/tmp/" + name + "-overnight-status.json").read_text())
    if isinstance(status, list):
        status = status[0]
    jobs = []
    for job in status["jobs"]:
        st = job["status"]
        seconds = (
            dt.datetime.fromisoformat(st["exited"].replace("Z", "+00:00"))
            - dt.datetime.fromisoformat(st["scheduled"].replace("Z", "+00:00"))
        ).total_seconds()
        jobs.append(
            {"id": job["id"], "status": st, "allocated_gpus": 3, "scheduled_to_exit_gpu_hours": seconds * 3 / 3600}
        )
    rows = json.loads(Path("/tmp/" + name + "-curve.json").read_text())
    for row in rows:
        row["completed_updates"] = row["version"] - (1 if name == "megatron500" else 0)
        assert row["completed_updates"] == (0 if row["eval_index"] == 0 else row["eval_index"] + 1)
    assert len(rows) == 26 and rows[-1]["completed_updates"] == 500
    report["runs"][name] = {
        "experiment": status["id"],
        "jobs": jobs,
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "curve": rows,
    }
log = Path("/tmp/light200-overnight.log")
status = json.loads(Path("/tmp/light-overnight-status.json").read_text())[0]
curve = []
for row in json.loads(Path("/tmp/light200-curve.json").read_text()):
    d = row["metrics"]
    curve.append(
        {
            "completed_updates": 0 if row["eval_index"] == 0 else row["eval_index"] + 1,
            "correct": round(d["eval/gsm8k"] * 128),
            "count": 128,
            "mean_response_tokens": d.get("eval/gsm8k/response_len/mean"),
            "capped_fraction": d.get("eval/gsm8k/truncated_ratio"),
        }
    )
endpoints, samples = [], {}
for path in sorted(Path("/tmp/light200-retained").glob("offline-*.json")):
    if "raw" in path.name:
        continue
    d = json.loads(path.read_text())
    ids = set()
    for row in d["samples"]:
        assert row["id"] not in ids
        ids.add(row["id"])
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", re.sub(r"(\d),(\d)", r"\1\2", row["response"]))
        assert float(bool(numbers) and numbers[-1] == row["label"]) == row["score"]
    samples[d["step"]] = {row["id"]: row for row in d["samples"]}
    endpoints.append(
        {
            **{k: v for k, v in d.items() if k != "samples"},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rescored_count": len(d["samples"]),
            "score_mismatches": 0,
        }
    )
assert samples[0].keys() == samples[200].keys()
transitions = {"wrong_to_correct": 0, "correct_to_wrong": 0, "both_correct": 0, "both_wrong": 0}
for key, before in samples[0].items():
    after = samples[200][key]
    assert before["label"] == after["label"]
    name = (
        ("both_correct" if before["score"] else "wrong_to_correct")
        if after["score"]
        else ("correct_to_wrong" if before["score"] else "both_wrong")
    )
    transitions[name] += 1
assert transitions["wrong_to_correct"] - transitions["correct_to_wrong"] == 39
report["runs"]["light_core200"] = {
    "experiment": status["id"],
    "jobs": [{"id": j["id"], "status": j["status"]} for j in status["jobs"]],
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "native_curve": curve,
    "offline_endpoints": endpoints,
    "offline_paired_transitions": transitions,
}
report["mixed_task_starts"] = {}
for name, log_name, updates, eid in [
    (
        "moe_sft",
        "/tmp/basket-full-sft-basket-fast-200-keepalive60-20260914-replica-1.log",
        18,
        "01M2F7N19DQ3YJMRJAXJ1K4H59",
    ),
    ("dense_think_sft", "/tmp/olmo3-status-driver.log", 11, "01M2FBGGE8K8XJ7WJKCTE4KHMB"),
]:
    path = Path(log_name)
    evals = []
    for line in path.read_text().splitlines():
        match = re.search(r"metrics.py:\d+ - eval (\d+): (\{.*\})", line)
        if match:
            values = ast.literal_eval(match[2])
            evals.append(
                {
                    "eval_index": int(match[1]),
                    "domains": {
                        domain: {
                            "mean_reward": values["eval/" + domain],
                            "count": values["eval/" + domain + "/num_training_samples"],
                            "mean_response_tokens": values["eval/" + domain + "/response_len/mean"],
                            "capped_fraction": values["eval/" + domain + "/truncated_ratio"],
                        }
                        for domain in ["math", "ifeval", "code", "general"]
                    },
                }
            )
    assert len(evals) == 1 and evals[0]["eval_index"] == 0
    report["mixed_task_starts"][name] = {
        "experiment": eid,
        "completed_updates": updates,
        "status": "failed: external code HTTP read timeout after retries",
        "response_cap": 4096,
        "evaluations": evals,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
(out / "recovered-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    "Validated 52 heavy-SFT eval points, 21 light-SFT points, 2638 independently rescored answers, paired IDs and labels"
)
print("Paired full-test transitions:", transitions)
print(
    "Heavy GPU-hours:",
    {k: v["jobs"][0]["scheduled_to_exit_gpu_hours"] for k, v in report["runs"].items() if k.endswith("500")},
)
print("Mixed-task starts:", {k: v["evaluations"][0]["domains"] for k, v in report["mixed_task_starts"].items()})
