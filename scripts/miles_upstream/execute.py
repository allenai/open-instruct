"""Execute a frozen upstream plan, checking its schema and source identity first."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from open_instruct.miles.configuration import options
from open_instruct.miles.errors import InputError


def verify(plan, root=Path("/root/miles"), schemas=Path("/opt/miles-upstream/schemas")):
    if plan["backend"] not in ("fsdp", "megatron", "torchtitan"):
        raise InputError("Invalid upstream backend")
    schema_path = schemas / f"{plan['backend']}.json"
    schema = json.loads(schema_path.read_text())
    if plan["build_id"] != schema["build_id"] or plan["source"] != schema["source"]:
        raise InputError("Submitted plan and runtime have different image/source identities")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != plan["source"]["revision"]:
        raise InputError("Runtime MILES checkout differs from the validated source")
    argv = options.encode_options(plan["native"], schema_path=schema_path)
    entrypoint = "train_async.py" if plan["native"].get("fully_async", False) else "train.py"
    if (
        plan["native"].get("train_backend") != plan["backend"]
        or argv != plan["argv"]
        or entrypoint != plan["entrypoint"]
    ):
        raise InputError("Submitted native arguments differ from the validated plan")
    return [sys.executable, str(root / entrypoint), *argv]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--schema-dir", type=Path, default=Path("/opt/miles-upstream/schemas"))
    args = parser.parse_args()
    command = verify(json.loads(args.plan.read_text()), schemas=args.schema_dir)
    os.chdir("/root/miles")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
