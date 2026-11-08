"""Run cache correctness, fixed-token serving, then matched training on one node."""

import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from open_instruct.miles.execution import launch


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def logged_run(command, log, *, timeout, env=None):
    print("CACHE_PANEL_PHASE", str(log), flush=True)
    pipeline = shlex.join(command) + " 2>&1 | tee " + shlex.quote(str(log))
    subprocess.run(["bash", "-o", "pipefail", "-c", pipeline], env=env, check=True, timeout=timeout)


def parity(a, b):
    same = a["tokens"] == b["tokens"]
    error = max((abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"], strict=True)), default=0) if same else None
    return dict(same_tokens=same, max_logprob_error=error, passed=same and error <= 1e-4)


def serving(root, mode, arm, model, data):
    directory = root / (mode + "-" + arm)
    command = [sys.executable, "-m", "scripts.miles.cache_panel.serving", mode, "--output", str(directory)]
    if arm == "on":
        command.append("--cached")
    if mode == "benchmark":
        command += ["--model", model, "--data", data]
    directory.mkdir(parents=True, exist_ok=True)
    logged_run(command, directory / "console.log", env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, timeout=1800)
    return read(directory / "report.json")


def check_serving(root, model, data):
    tiny = {arm: serving(root, "tiny", arm, model, data) for arm in ("off", "on")}
    checks = [parity(tiny["off"]["cold"], tiny["on"]["cold"])]
    for a, b in zip(tiny["off"]["rounds"], tiny["on"]["rounds"], strict=True):
        checks.append(parity(a["fresh"], b["fresh"]))
    write(root / "tiny-parity.json", checks)
    assert all(r["passed"] for r in checks), "Fresh-policy tiny-model numerical parity failed"
    reports = {arm: serving(root, "benchmark", arm, model, data) for arm in ("off-before", "on", "off-after")}
    comparisons = []
    for before, cached, after in zip(*(reports[a]["waves"] for a in ("off-before", "on", "off-after")), strict=True):
        control = [parity(a, b) for a, b in zip(before["responses"], after["responses"], strict=True)]
        candidate = [parity(a, b) for a, b in zip(before["responses"], cached["responses"], strict=True)]
        comparisons.append(
            dict(
                index=cached["index"],
                repeat=cached["repeat"],
                control=control,
                candidate=candidate,
                seconds={"before": before["seconds"], "cached": cached["seconds"], "after": after["seconds"]},
                cached_tokens=sum(r["cached_tokens"] for r in cached["responses"]),
            )
        )
    write(root / "serving-comparison.json", comparisons)
    assert all(r["passed"] for c in comparisons for r in c["control"]), (
        "Cache-off controls disagree; investigate before interpreting candidate"
    )
    assert all(r["passed"] for c in comparisons for r in c["candidate"]), (
        "Candidate numerical parity requires investigation"
    )
    assert sum(c["cached_tokens"] for c in comparisons) > 0, "No prefix reuse observed"


def training_summary(directory):
    checkpoint = directory / "checkpoints"
    contracts = [json.loads(line) for line in (checkpoint / "training_contract_rank0.jsonl").read_text().splitlines()]
    refresh = [r for r in contracts if r["event"] == "refresh_scores"]
    optimizer = [r for r in contracts if r["event"] == "optimizer"]
    timing = [json.loads(line) for line in (checkpoint / "driver_timing.jsonl").read_text().splitlines()]
    samples = []
    for path in checkpoint.glob("sibling_timing_*.jsonl"):
        for line in path.read_text().splitlines():
            samples += json.loads(line)["samples"]
    measured = [
        r
        for r in timing
        if r.get("rollout_id", -1) is not None
        and r.get("rollout_id", -1) >= 4
        and r["stage"] in ("training", "publication", "generation_wait")
    ]
    per_update = {}
    for r in measured:
        per_update[r["rollout_id"]] = per_update.get(r["rollout_id"], 0) + r["seconds"]
    known = [s for s in samples if "engine_reported_cached_tokens" in s]
    return dict(
        updates=len(optimizer),
        consumed_cross_version_responses=sum(r["mixed_responses"] for r in refresh),
        consumed_behavior_versions=[r["local_behavior_versions"] for r in optimizer],
        steady_update_seconds=per_update,
        steady_mean_seconds=statistics.mean(per_update.values()) if per_update else None,
        generated_response_records=len(samples),
        cache_observation_coverage=len(known),
        cached_tokens=sum(s["engine_reported_cached_tokens"] for s in known),
        prompt_tokens=sum(s.get("engine_reported_prompt_tokens", 0) for s in known),
        response_tokens=sum(s.get("response_tokens", 0) for s in samples),
        hits=sum(s["engine_reported_cached_tokens"] > 0 for s in known),
        retractions=sum(s.get("engine_reported_num_retractions", 0) for s in samples),
    )


def main():
    root = Path("/output/cache-panel")
    root.mkdir(parents=True, exist_ok=True)
    specs = Path("/opt/core-rl/cache-panel-data")
    base = read(specs / "off.json")
    state = {"status": "serving_checks", "training": {}}
    write(root / "status.json", state)
    try:
        check_serving(root, base["model"]["source"], base["data"]["prompt_data"])
        for arm in ("off", "on"):
            state["status"] = "training_" + arm
            write(root / "status.json", state)
            subprocess.run(["ray", "stop", "--force"], check=False, timeout=90)
            directory = Path(read(specs / (arm + ".json"))["output"]["root"])
            try:
                logged_run(
                    [sys.executable, "-m", "open_instruct.miles", "train", str(specs / (arm + ".json"))],
                    root / ("training-" + arm + ".log"),
                    timeout=4500,
                )
            finally:
                launch.collect_results(str(directory), str(root / ("training-" + arm)))
            state["training"][arm] = training_summary(directory)
            write(root / "status.json", state)
        state["status"] = "complete"
        state["coverage_passed"] = all(
            x["updates"] == 24 and x["consumed_cross_version_responses"] > 0 for x in state["training"].values()
        )
        assert state["coverage_passed"], "Training completed but required mixed-policy consumption was not exercised"
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        raise
    finally:
        write(root / "status.json", state)


if __name__ == "__main__":
    main()
