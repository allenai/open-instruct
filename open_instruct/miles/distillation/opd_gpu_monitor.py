"""Five-second whole-device snapshots; utilization is sampled, not an idle-time counter."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

FIELDS = ["index", "uuid", "utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw"]


def parse(text, roles):
    output = []
    for values in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(values) != len(FIELDS):
            raise ValueError("Unexpected nvidia-smi field count")
        index = int(values[0])
        row = dict(gpu=index, uuid=values[1], role=next((k for k, v in roles.items() if index in v), "unknown"))
        for key, value in zip(FIELDS[2:], values[2:], strict=True):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = None
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", required=True)
    args = parser.parse_args()
    roles = json.loads(args.roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as stream:
        while True:
            row = dict(time_unix=time.time(), memory_unit="MiB", utilization_unit="percent")
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=4,
                )
                row["gpus"] = parse(result.stdout, roles)
            except Exception as exc:
                row["error"] = type(exc).__name__
            stream.write(json.dumps(row) + "\n")
            time.sleep(5)


if __name__ == "__main__":
    main()
