"""Run the pinned local contract suite and retain numerical evidence."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

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
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        f"--junitxml={args.output / 'junit.xml'}",
    ]
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    measurements = []
    for line in log.read_text().splitlines():
        for marker in ("CONTRACT_MEASUREMENT ", "CORE_RESUME_CONTRACT "):
            if marker in line:
                measurements.append({"kind": marker.strip(), "value": json.loads(line.split(marker, 1)[1])})
    files = sorted(Path("open_instruct/miles").glob("*.py")) + [
        Path("tests/miles/test_contract.py"),
        Path("tests/miles/test_runtime.py"),
    ]
    report = dict(
        passed=result.returncode == 0,
        exit_code=result.returncode,
        elapsed_seconds=time.monotonic() - start,
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        measurements=measurements,
        limits=[
            "Gloo reference tests do not qualify native EP; use ep_contract.py on two GPUs",
            "No full-SFT replay or matched Megatron comparison in this suite",
        ],
    )
    (args.output / "contract.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "exit_code", "elapsed_seconds", "device")}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
