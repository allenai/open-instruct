"""Capture every TPU pod's logs and enforce registered mechanical checks.

Does not launch or restart jobs. A numerical failure deactivates only the named
JobSet's Kueue workload. Missing evidence is never a pass. Configuration is a
JSON file naming the job, expected steps/batches and an optional HF reference.
"""

import argparse
import concurrent.futures
import json
import math
import re
import subprocess
import time
from pathlib import Path

KUBECTL = "/snap/google-cloud-cli/current/bin/kubectl"


def unique_insert(table, key, value):
    assert key not in table or table[key] == value, f"Conflicting duplicated record: {key}"
    table[key] = value


def parse_logs(text):
    parts, batches, drops, restores, skips = {}, {}, {}, {}, set()
    for step, ce, lb, z, tokens in re.findall(
        r"LOSS-PARTS step=(\d+) ce=(\S+) moe_lb=(\S+) z=(\S+) tokens=(\d+)", text
    ):
        unique_insert(parts, int(step), {"ce": float(ce), "moe_lb": float(lb), "z": float(z), "tokens": int(tokens)})
    for step, host, sha in re.findall(r"BATCH-HASH n=(\d+) host=(\d+) sha256=([0-9a-f]{64})", text):
        unique_insert(batches, (int(host), int(step)), sha)
    for step, dropped, routed in re.findall(r"MOE-DROPPED step=(\d+) dropped=(\d+) routed=(\d+)", text):
        unique_insert(drops, int(step), {"dropped": int(dropped), "routed": int(routed)})
    for line in text.splitlines():
        if "H127-RESUME-RESTORE " in line:
            value = json.loads(line.split("H127-RESUME-RESTORE ", 1)[1])
            unique_insert(restores, value["process"], value)
    skips.update(map(int, re.findall(r"Resume: skipped (\d+) already-trained batches", text)))
    return parts, batches, drops, restores, skips


def assess(config, text, terminal=False):
    parts, batches, drops, restores, skips = parse_logs(text)
    expected = {(row["host"], row["step"]): row["sha256"] for row in config["expected_batches"]}
    mismatches = [list(key) for key, sha in expected.items() if key in batches and batches[key] != sha]
    failures, missing = [], []
    if mismatches:
        failures.append("Batch identity mismatch")
    if any(not math.isfinite(v[k]) for v in parts.values() for k in ("ce", "moe_lb", "z")):
        failures.append("Nonfinite loss")
    gap = None
    if "reference_ce" in config and 1 in parts:
        gap = abs(parts[1]["ce"] - config["reference_ce"])
        if gap > 1e-3 or parts[1]["tokens"] != config["reference_tokens"]:
            failures.append("Step-one HF CE/token-count mismatch")
    if any(d["routed"] <= 0 or d["dropped"] < 0 or d["dropped"] > d["routed"] for d in drops.values()):
        failures.append("Invalid routing-drop counts")
    max_drop = max((d["dropped"] / max(d["routed"], 1) for d in drops.values()), default=0.0)
    drop_fraction = sum(d["dropped"] for d in drops.values()) / max(1, sum(d["routed"] for d in drops.values()))
    if max_drop > 0.01:
        failures.append("Per-step routing-drop budget exceeded")
    expected_steps = set(range(config.get("first_step", 1), config["steps"] + 1))
    if set(parts) - expected_steps:
        failures.append("Unexpected trained step (possible restart from zero)")
    resume_step = config.get("resume_step")
    if resume_step is not None:
        if any(v["restored_step"] != resume_step or v["optimizer_step"] != resume_step for v in restores.values()):
            failures.append("Wrong restored step")
        if skips - {resume_step}:
            failures.append("Wrong data-resume cursor")
    if terminal:
        if set(parts) != expected_steps:
            missing.append("Incomplete loss steps")
        if set(drops) != expected_steps:
            missing.append("Incomplete routing-drop steps")
        if not set(expected) <= set(batches):
            missing.append("Incomplete registered batch hashes")
        if drop_fraction > 1e-4:
            failures.append("Whole-run routing-drop budget exceeded")
        if resume_step is not None and (set(restores) != set(range(32)) or skips != {resume_step}):
            missing.append("Incomplete 32-host restore audit or data-skip evidence")
    return {
        "last_step": max(parts, default=0),
        "step1_ce_gap": gap,
        "failures": failures,
        "missing": missing,
        "expected_hashes": len(expected),
        "matched_hashes": len(set(expected) & set(batches)) - len(mismatches),
        "batch_hash_mismatches": mismatches,
        "loss_parts": parts,
        "drops": drops,
        "drop_fraction": drop_fraction,
        "max_step_drop_fraction": max_drop,
        "restore_audits": restores,
        "resume_skips": sorted(skips),
        "passed": terminal and not failures and not missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    job = config["job"]
    assert job.startswith("robertb-pres14t-v1-")
    args.output.mkdir(exist_ok=True)
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    seen = {p.stem: set(p.read_text().splitlines()) for p in logs.glob("*.log")}

    def kube(*cmd):
        return subprocess.check_output([KUBECTL, "-n", "olmo", *cmd], text=True, timeout=120)

    def stop():
        workloads = json.loads(kube("get", "workloads", "-o", "json"))["items"]
        for item in workloads:
            if any(
                r.get("kind") == "JobSet" and r.get("name") == job for r in item["metadata"].get("ownerReferences", [])
            ):
                kube("patch", "workload", item["metadata"]["name"], "--type=merge", "-p", '{"spec":{"active":false}}')

    def capture(pod):
        name = pod["metadata"]["name"]
        try:
            text = kube("logs", name, "-c", "maxtext", "--timestamps=true", "--pod-running-timeout=5s")
        except (subprocess.SubprocessError, OSError) as error:
            return {"pod": name, "error": str(error)}
        previous = seen.setdefault(name, set())
        with (logs / f"{name}.log").open("a") as stream:
            for line in text.splitlines():
                if line not in previous:
                    stream.write(line + "\n")
                    previous.add(line)
        return {"pod": name, "captured": True}

    # The JobSet owns its execution deadline. A watcher observation timeout does
    # not cancel/relaunch running work: leave all captured evidence for review.
    started = time.monotonic()
    while time.monotonic() - started < config.get("watch_seconds", 8 * 3600):
        try:
            pods = json.loads(kube("get", "pods", "-l", f"jobset.sigs.k8s.io/jobset-name={job}", "-o", "json"))[
                "items"
            ]
        except (subprocess.SubprocessError, OSError) as error:
            print("OBSERVATION-ERROR", str(error), flush=True)
            time.sleep(30)
            continue
        (args.output / "pods.json").write_text(json.dumps(pods, indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            capture_results = list(pool.map(capture, pods))
        succeeded = sum(p["status"]["phase"] == "Succeeded" for p in pods)
        failed = sum(p["status"]["phase"] == "Failed" for p in pods)
        text = "\n".join(p.read_text() for p in sorted(logs.glob("*.log")))
        try:
            result = assess(config, text, terminal=succeeded == 32)
        except (AssertionError, ValueError):
            stop()
            raise
        result.update(job=job, succeeded_pods=succeeded, failed_pods=failed, captures=capture_results)
        temporary = args.output / "receipt.json.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.output / "receipt.json")
        print(
            json.dumps(
                {k: result[k] for k in ("job", "last_step", "failures", "missing", "matched_hashes", "passed")}
            ),
            flush=True,
        )
        if result["failures"] or failed:
            stop()
            raise RuntimeError("TPU run failed; workload deactivated, no automatic restart")
        if succeeded == 32:
            assert result["passed"], result["missing"]
            return
        time.sleep(30)
    raise TimeoutError("Watcher expired; inspect existing workload, never infer it stopped")


if __name__ == "__main__":
    main()
