"""Persist latest-attempt Beaker status and summarize completed H235 noise probes."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from scripts.miles.diagnostics import summarize_gradient_noise


def command(*arguments):
    for attempt in range(5):
        try:
            return subprocess.check_output(["beaker", *arguments], text=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def latest_jobs(experiment):
    jobs = json.loads(command("experiment", "get", experiment, "--format", "json"))[0]["jobs"]
    latest = {}
    for job in jobs:
        key = job["execution"]["task"], job["execution"].get("replicaRank", 0)
        if key not in latest or job["status"]["created"] > latest[key]["status"]["created"]:
            latest[key] = job
    return list(latest.values())


def monitor(runs, output, *, once=False):
    output.mkdir(parents=True, exist_ok=True)
    while True:
        states = {}
        for stage, experiment in runs.items():
            jobs = latest_jobs(experiment)
            state = {"experiment": experiment, "jobs": [{"id": j["id"], "status": j["status"]} for j in jobs]}
            if jobs and all(j["status"].get("finalized") for j in jobs):
                if any(j["status"].get("exitCode") != 0 for j in jobs):
                    state["state"] = "failed"
                    for job in jobs:
                        (output / (stage + "-failure.log")).write_text(
                            command("job", "logs", job["id"], "--tail", "200")
                        )
                else:
                    target = output / stage
                    if not (target / "summary.json").exists():
                        # The result dataset is owned by the successful latest job.
                        dataset = next(j["result"]["beaker"] for j in jobs if j.get("result"))
                        command("dataset", "fetch", dataset, "--prefix", "run/gradient-noise.jsonl", "-o", str(target))
                        rows = [
                            json.loads(line) for line in (target / "run/gradient-noise.jsonl").read_text().splitlines()
                        ]
                        if len(rows) != 8 or sorted(row["step"] for row in rows) != list(range(1, 9)):
                            raise ValueError("Expected exactly eight distinct successful batch measurements")
                        if (
                            len({row["parameter_elements"] for row in rows}) != 1
                            or rows[0]["parameter_elements"] < 1e9
                        ):
                            raise ValueError("Missing or inconsistent full-model gradient coverage")
                        (target / "summary.json").write_text(
                            json.dumps(summarize_gradient_noise.summarize(rows), indent=2) + "\n"
                        )
                    state["state"] = "complete"
                    state["summary"] = json.loads((target / "summary.json").read_text())
            else:
                state["state"] = "running" if any(j["status"].get("started") for j in jobs) else "queued"
                if state["state"] == "queued":
                    for job in jobs:
                        state.setdefault("scheduler_events", {})[job["id"]] = command("job", "events", job["id"])
                else:
                    for job in jobs:
                        logs = command("job", "logs", job["id"], "--tail", "500")
                        rows = [line for line in logs.splitlines() if "Half-group gradient noise:" in line]
                        state.setdefault("recent_measurements", []).extend(rows)
            states[stage] = state
        (output / "status.json").write_text(json.dumps(states, indent=2) + "\n")
        print(
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            {k: v["state"] for k, v in states.items()},
            flush=True,
        )
        if once or all(v["state"] in {"failed", "complete"} for v in states.values()):
            return
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    options = parser.parse_args()
    while True:
        try:
            monitor({"sft": options.sft, "final512": options.final}, options.output, once=options.once)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"Transient Beaker failure; retrying in 60s: {error}", flush=True)
            if options.once:
                raise
            time.sleep(60)


if __name__ == "__main__":
    main()
