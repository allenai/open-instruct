"""Run the pinned local contract suite and retain numerical evidence."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "pytest.log"
    start = time.monotonic()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/miles/test_contract.py",
        "tests/miles/test_runtime.py",
        "tests/miles/test_lifecycle.py",
        "tests/miles/test_contract_early_validation.py",
        "tests/miles/test_ep_contract_comparison.py",
        "tests/miles/test_datasource_trials.py",
        "open_instruct/test_miles_reward_process.py",
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        f"--junitxml={args.output / 'junit.xml'}",
    ]
    files = (
        sorted(Path("open_instruct/miles").glob("*.py"))
        + [Path(value) for value in command if value.endswith(".py")]
        + sorted(Path("scripts/miles").glob("*.py"))
        + sorted(Path("configs/miles").rglob("*.toml"))
        + [Path("tests/miles/ep_contract.py"), Path("runtime/miles/runtime.lock.json")]
    )
    source_sha256 = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    measurements = []
    for line in log.read_text().splitlines():
        for marker in ("CONTRACT_MEASUREMENT ", "CORE_RESUME_CONTRACT ", "AUXILIARY_CONTRACT ", "REPLAY_CONTRACT "):
            if marker in line:
                measurements.append({"kind": marker.strip(), "value": json.loads(line.split(marker, 1)[1])})
    changed_sources = [
        str(path) for path in files if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256[str(path)]
    ]
    suites = ElementTree.parse(args.output / "junit.xml").getroot().iter("testsuite")
    test_counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in test_counts:
            test_counts[key] += int(suite.get(key, "0"))
    report = dict(
        test_counts=test_counts,
        passed=result.returncode == 0 and not changed_sources,
        changed_sources=changed_sources,
        exit_code=result.returncode,
        elapsed_seconds=time.monotonic() - start,
        runtime_lock=json.loads(Path("runtime/miles/runtime.lock.json").read_text()),
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        source_sha256=source_sha256,
        measurements=measurements,
        limits=[
            "Gloo reference tests do not qualify native EP; use ep_contract.py on two GPUs",
            "No full-SFT replay or matched Megatron comparison in this suite",
        ],
    )
    (args.output / "contract.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "exit_code", "elapsed_seconds", "device")}))
    raise SystemExit(result.returncode or int(bool(changed_sources)))


if __name__ == "__main__":
    main()
