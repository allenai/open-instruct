"""Opt-in local Miles integration group; never collected by ordinary pytest.

Run inside the candidate runtime with one visible GPU. No cluster submission.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def execute(command, output, env, timeout):
    started = time.monotonic()
    with output.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"Timed out after {timeout}s; see {output}") from None
    if code:
        raise RuntimeError(f"Command exited {code}; see {output}")
    return {"command": command, "log": str(output), "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("lifecycle", "resume", "all"), default="all")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--miles-source", required=True, type=Path, help="Expected imported candidate checkout")
    parser.add_argument(
        "--image-id", required=True, help="Immutable image ID/digest supplied by the container launcher"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-process limit in seconds")
    options = parser.parse_args()
    root = options.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OLMO_SGLANG_CORE_COMPAT="rounding", OLMO_SGLANG_ROUNDING_KERNELS="torch")
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="17", HF_HUB_OFFLINE="1")
    report = {"passed": False, "case": options.case, "commands": []}

    def call(script, *arguments):
        command = [sys.executable, str(script), *map(str, arguments)]
        log = root / f"{len(report['commands']):02d}-{script.stem}.log"
        report["commands"].append(execute(command, log, env, options.timeout))

    def train(name, updates, *, resume=False, export=True):
        arguments = [
            "run",
            root / name,
            "--updates",
            updates,
            "--greedy",
            "--routing-replay",
            "--diagnostic-interval",
            1,
        ]
        if export:
            arguments.append("--export-hf")
        if resume:
            arguments.append("--resume")
        call(HERE.parent / "local_moe.py", *arguments)

    try:
        call(
            HERE / "runtime.py",
            "prepare",
            root,
            "--miles-source",
            options.miles_source.resolve(),
            "--image-id",
            options.image_id,
        )
        # Three uninterrupted updates are also the lifecycle fixture for the resume pair.
        train("control", 3)
        call(HERE / "runtime.py", "audit", root / "control")
        if options.case in ("resume", "all"):
            train("resumed", 2, export=False)
            train("resumed", 3, resume=True)
            call(HERE / "runtime.py", "audit", root / "resumed")
            call(HERE / "runtime.py", "compare", root)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MILES_INTEGRATION_PASSED: {root / 'result.json'}")


if __name__ == "__main__":
    main()
