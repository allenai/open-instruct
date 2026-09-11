"""Observe authorized learning/diagnostic runs; audit once after both500 arms succeed."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

EXPERIMENTS = {
    "core": "01M279ZFM6RBC223RJJ6QHN9MP",
    "megatron": "01M278B5E9HME181B04HT6391P",
    "light": "01M2794XP15PHQSWN6M499NQYS",
    "score_variants": "01M27BKYTF9N7JMYBTKAV0HS9A",
}
POLL_SECONDS = 300
MAX_SECONDS = 24 * 3600


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def execute(argv, **kwargs):
    timeout = kwargs.pop("timeout", 300)
    with subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        **kwargs,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv, stdout, stderr)
        return stdout


def submitted_id(output):
    """The standard wrapper ends with the single experiment-create JSON array."""
    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(output):
        if character != "[":
            continue
        try:
            value, end = decoder.raw_decode(output[index:])
        except ValueError:
            continue
        if (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], dict)
            and value[0].get("id")
            and value[0].get("workspaceRef")
            and not output[index + end :].strip()
        ):
            candidates.append(value[0]["id"])
    if len(candidates) != 1:
        raise ValueError("Submission result is ambiguous; never automatically submit again")
    return candidates[0]


def allocation_seconds(job):
    status = job["status"]
    return (
        dt.datetime.fromisoformat(status["exited"].replace("Z", "+00:00"))
        - dt.datetime.fromisoformat(status["scheduled"].replace("Z", "+00:00"))
    ).total_seconds()


class Watcher:
    def __init__(self, output, config, runner=execute):
        self.output, self.config, self.runner = Path(output), config, runner
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "state.json"
        self.state = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"config": config, "created": now(), "jobs": {}, "audit": {}, "attention": []}
        )
        if self.state["config"] != config:
            raise ValueError("Watcher configuration differs from its persisted identity")
        if self.state["audit"].get("status") == "submitting":
            self.state["audit"]["status"] = "ambiguous"
            self.attention("Watcher interrupted during audit submission; inspect submission.log manually")
        if self.state.get("analysis") == "attempted":
            self.attention("Local analysis was interrupted or failed; audit will not be resubmitted")
        self.save()

    def save(self):
        atomic_json(self.path, self.state)

    def attention(self, message):
        if message not in self.state["attention"]:
            self.state["attention"].append(message)
        self.save()

    def call(self, argv, **kwargs):
        remaining = dt.datetime.fromisoformat(self.state["created"]).timestamp() + MAX_SECONDS - time.time()
        if remaining <= 0:
            raise TimeoutError("24-hour watcher deadline reached")
        return self.runner(argv, timeout=min(kwargs.pop("timeout", 300), remaining), **kwargs)

    def observe(self, name, experiment):
        data = json.loads(self.call(["beaker", "experiment", "get", experiment, "--format", "json"]))
        if len(data) != 1 or data[0]["id"] != experiment or len(data[0].get("jobs", [])) != 1:
            raise ValueError(f"Unexpected experiment identity or job count for {name}")
        raw_job = data[0]["jobs"][0]
        # Persist status/provenance, never environment values or secret references.
        job = {key: raw_job[key] for key in ("id", "status", "result", "node", "requests", "limits") if key in raw_job}
        record = self.state["jobs"].setdefault(name, {"experiment": experiment})
        if record.get("job_id", job["id"]) != job["id"]:
            self.attention(f"{name}: job identity changed; automatic restart/retry is outside this watcher")
            return
        record["job_id"] = job["id"]
        if record.get("status") != job["status"]:
            directory = self.output / name
            directory.mkdir(exist_ok=True)
            encoded = json.dumps(job, sort_keys=True).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            snapshot = directory / f"status-{digest}.json"
            if not snapshot.exists():
                snapshot.write_bytes(encoded + b"\n")
            record["status"] = job["status"]
            record["snapshot"] = str(snapshot.relative_to(self.output))
        record["job"] = job
        self.save()
        if job["status"].get("finalized") and job["status"].get("exitCode") != 0:
            self.attention(f"{name}: exited {job['status'].get('exitCode')}; no automatic retry")

    def capture_terminal(self, name):
        record = self.state["jobs"].get(name, {})
        if not self.complete(name) or record.get("captured"):
            return
        directory = self.output / name
        committed = directory / "capture/complete.json"
        if committed.is_file():
            proof = json.loads(committed.read_text())
            if proof["job_id"] != record["job_id"]:
                raise ValueError("Captured job identity differs from terminal metadata")
            record.update(proof, captured=True)
            self.save()
            return
        if record.get("capture_attempts", 0) >= 3:
            self.attention(f"{name}: terminal capture exhausted three attempts; inspect capture staging directories")
            return
        attempt = record["capture_attempts"] = record.get("capture_attempts", 0) + 1
        self.save()
        staging = directory / f"capture-attempt-{attempt}"
        try:
            staging.mkdir()
            log = self.call(["beaker", "job", "logs", record["job_id"]])
            (staging / "final.log").write_text(log)
            dataset = record["job"]["result"]["beaker"]
            self.call(["beaker", "dataset", "fetch", dataset, "--output", str(staging / "results")])
            proof = {
                "job_id": record["job_id"],
                "result_id": dataset,
                "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
            }
            atomic_json(staging / "complete.json", proof)
            staging.rename(directory / "capture")
            record.update(proof, captured=True)
        except Exception as error:
            failure = {"attempt": attempt, "at": now(), "error": str(error)}
            record.setdefault("capture_errors", []).append(failure)
            if staging.is_dir():
                atomic_json(staging / "capture-error.json", failure)
            if attempt == 3:
                self.attention(
                    f"{name}: terminal capture exhausted three attempts; inspect capture staging directories"
                )
        self.save()

    def complete(self, name):
        return bool(self.state["jobs"].get(name, {}).get("status", {}).get("finalized"))

    def successful(self, name):
        return self.complete(name) and self.state["jobs"][name]["status"].get("exitCode") == 0

    def verify_checkout(self):
        checkout = self.config["checkout"]
        if self.call(["git", "rev-parse", "HEAD"], cwd=checkout).strip() != self.config["source"]:
            raise ValueError("Pinned checkout commit changed")
        if self.call(["git", "status", "--porcelain"], cwd=checkout).strip():
            raise ValueError("Pinned checkout is dirty")

    def submit_audit(self):
        if self.config.get("observe_only"):
            raise ValueError("Observe-only watchers cannot submit audits")
        self.state["audit"] = {"status": "submitting", "attempted_at": now()}
        self.save()  # The durable intent precedes every possible external side effect.
        command = [
            "./scripts/train/build_image_and_launch.sh",
            "--miles",
            "scripts/train/debug/miles_core_gsm8k_extended.sh",
            "--stage",
            "audit",
        ]
        try:
            self.verify_checkout()
            output = self.call(
                command,
                cwd=self.config["checkout"],
                env={
                    **{key: value for key, value in os.environ.items() if key != "MILES_EXISTING_IMAGE"},
                    "MILES_BASE_IMAGE": self.config["base_image"],
                },
                timeout=1800,
            )
            (self.output / "submission.log").write_text(output)
            experiment = submitted_id(output)
            self.state["audit"].update(status="submitted", experiment=experiment)
        except Exception as error:
            details = []
            for value in (getattr(error, "stdout", None), getattr(error, "stderr", None)):
                details.append(value.decode(errors="replace") if isinstance(value, bytes) else (value or ""))
            details = "\n".join(details)
            (self.output / "submission.log").write_text(details + "\n" + str(error))
            self.state["audit"]["status"] = "ambiguous"
            self.attention("Audit submission ambiguous or failed; inspect submission.log before any manual action")
        self.save()

    def analyze(self):
        self.state["analysis"] = "attempted"
        self.save()
        try:
            self.verify_checkout()
            source = self.output / "audit/capture/results"
            target = self.output / "analysis"
            target.mkdir(exist_ok=True)
            for backend in ("core", "megatron"):
                path = source / f"{backend}-audit.json"
                if not json.loads(path.read_text()).get("valid"):
                    raise ValueError(f"{backend} independent audit did not pass")
                destination = target / backend
                destination.mkdir(exist_ok=True)
                shutil.copyfile(path, destination / "audit.json")
            if not json.loads((source / "comparison.json").read_text()).get("valid"):
                raise ValueError("Paired CPU comparison did not pass")
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                "--mount",
                f"type=bind,src={self.output},dst=/watch",
                "--mount",
                f"type=bind,src={self.config['checkout']}/scripts/miles,dst=/opt/core-rl/scripts/miles,readonly",
                self.config["analysis_image"],
                "scripts/miles/analyze_gsm8k_parity.py",
                "compare",
                "/watch/analysis",
                "--updates",
                "500",
                "--eval-interval",
                "20",
                "--core-log",
                "/watch/core/capture/final.log",
                "--megatron-log",
                "/watch/megatron/capture/final.log",
                "--output",
                "/watch/analysis/comparison.json",
                "--plot",
                "/watch/analysis/comparison.png",
            ]
            for backend in ("core", "megatron"):
                command.extend(
                    [f"--{backend}-allocated-seconds", str(allocation_seconds(self.state["jobs"][backend]["job"]))]
                )
            output = self.call(command, timeout=600)
            (target / "analyzer.log").write_text(output)
            report = json.loads((target / "comparison.json").read_text())
            if not report.get("valid") or not (target / "comparison.png").is_file():
                raise ValueError("Final comparison or plot is incomplete")
            self.write_summary(report, target / "comparison.md")
            self.state["analysis"] = "complete"
        except Exception as error:
            self.attention(f"Local comparison needs attention: {error}")
        self.save()

    def write_summary(self, report, path):
        lines = [
            "# Completed 500-update comparison",
            "",
            "Both runs and independent audits passed. One seed per backend; these are descriptive results.",
            "",
            "This fresh 500-update study starts from the SFT checkpoint. It is neither a resume nor a pure horizon extension: Core includes the scorer correction and serving settings changed. Measure gains against each new run's own initial evaluation.",
            "",
            f"[Configuration differences]({Path(self.config['checkout']) / 'docs/measurements/miles-gsm8k-configuration-differences-20260911.md'}) and [campaign protocol]({Path(self.config['checkout']) / 'docs/miles-learning-comparisons-20260911.md'}).",
            "",
            "| Updates | Core correct /128 | Megatron correct /128 |",
            "|---:|---:|---:|",
        ]
        for row in report["learning_curves"]:
            lines.append(
                f"| {row['completed_steps']} | {round(row['core']['accuracy'] * 128)} | {round(row['megatron']['accuracy'] * 128)} |"
            )
        lines += [
            "",
            "Allocated time uses Beaker scheduled→exited timestamps and includes startup, evaluation, and saves.",
            "",
        ]
        for backend, seconds in report["allocated_runtime_seconds"].items():
            lines.append(f"- {backend}: {seconds / 3600:.3f} allocated hours; {3 * seconds / 3600:.3f} GPU-hours.")
        cycles = report.get("comparable_timing", {}).get("collection_boundary_cycle", {})
        if cycles.get("indices"):
            lines += [
                "",
                f"Matched warm operational cycles: {len(cycles['indices'])}; Core {cycles['core']['mean_seconds']:.3f}s, Megatron {cycles['megatron']['mean_seconds']:.3f}s. Phase scopes differ; do not add independently averaged phase timers.",
            ]
        lines += [
            "",
            "[Full comparison JSON](comparison.json) and [plot](comparison.png). Raw final logs, terminal Beaker metadata, and independent audits are retained alongside this analysis. Light-SFT results are retained separately and use a different evaluation protocol.",
            "",
        ]
        path.write_text("\n".join(lines))

    def poll(self):
        for name, experiment in self.config["experiments"].items():
            if not self.complete(name):
                try:
                    self.observe(name, experiment)
                except Exception as error:
                    self.state["last_poll_error"] = {"at": now(), "arm": name, "error": str(error)}
            self.capture_terminal(name)
        if (
            not self.config.get("observe_only")
            and all(self.successful(name) for name in ("core", "megatron"))
            and not self.state["audit"]
        ):
            self.submit_audit()
        audit = self.state["audit"]
        if audit.get("status") == "submitted":
            try:
                if not self.complete("audit"):
                    self.observe("audit", audit["experiment"])
                self.capture_terminal("audit")
                if (
                    self.successful("audit")
                    and self.state["jobs"]["audit"].get("captured")
                    and not self.state.get("analysis")
                ):
                    if not all(self.state["jobs"][b].get("captured") for b in ("core", "megatron")):
                        raise ValueError("Final training artifacts missing; cannot produce timing comparison")
                    self.analyze()
            except Exception as error:
                self.state["last_poll_error"] = {"at": now(), "arm": "audit", "error": str(error)}
        self.state["last_poll"] = now()
        self.save()
        arms_done = all(self.complete(name) for name in self.config["experiments"])
        audit_done = audit.get("status") == "ambiguous" or self.complete("audit")
        failed_arm = any(self.complete(n) and not self.successful(n) for n in ("core", "megatron"))
        captured_names = [*self.config["experiments"]]
        if self.complete("audit"):
            captured_names.append("audit")
        captures_done = all(
            self.state["jobs"].get(name, {}).get("captured")
            or self.state["jobs"].get(name, {}).get("capture_attempts", 0) >= 3
            for name in captured_names
        )
        return arms_done and captures_done and (self.config.get("observe_only") or failed_arm or audit_done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--analysis-image", required=True, help="Immutable local Docker sha256 image ID")
    parser.add_argument("--base-image", required=True)
    parser.add_argument(
        "--light-experiment", required=True, help="Explicit approved light-run identity; never replaced automatically"
    )
    parser.add_argument("--score-experiment", default=EXPERIMENTS["score_variants"])
    parser.add_argument(
        "--observe-only", action="store_true", help="Capture only the explicit light experiment; never submit an audit"
    )
    parser.add_argument(
        "--previous-watch-output", type=Path, help="Preserved original watcher history for a light retry"
    )
    args = parser.parse_args()
    if not args.analysis_image.startswith("sha256:"):
        parser.error("Pin the local analyzer Docker image by sha256 ID")
    checkout, output = args.checkout.resolve(), args.output.resolve()
    if execute(["git", "status", "--porcelain"], cwd=checkout).strip():
        raise ValueError("Use a clean committed checkout")
    commit = execute(["git", "rev-parse", "HEAD"], cwd=checkout).strip()
    config = dict(
        checkout=str(checkout),
        source=commit,
        analysis_image=args.analysis_image,
        base_image=args.base_image,
        experiments={**EXPERIMENTS, "light": args.light_experiment, "score_variants": args.score_experiment},
    )
    if args.observe_only:
        config.update(observe_only=True, experiments={"light": args.light_experiment})
    if args.previous_watch_output:
        if not args.observe_only:
            parser.error("--previous-watch-output requires --observe-only")
        config["previous_watch_output"] = str(args.previous_watch_output.resolve())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watcher = Watcher(output, config)
        atomic_json(output / "process.json", {"pid": os.getpid(), "started": now(), "source": commit})
        deadline = dt.datetime.fromisoformat(watcher.state["created"]).timestamp() + MAX_SECONDS
        while time.time() < deadline:
            if watcher.poll():
                watcher.state["finished"] = now()
                watcher.save()
                raise SystemExit(bool(watcher.state["attention"]))
            time.sleep(min(POLL_SECONDS, max(0, deadline - time.time())))
        watcher.attention("24-hour watch deadline reached; existing jobs left untouched")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
