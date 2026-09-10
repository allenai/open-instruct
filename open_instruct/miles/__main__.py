"""Run with python -m open_instruct.miles {plan,validate,train} CONFIG.toml."""

import argparse
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

from open_instruct.miles.config import RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "train"))
    parser.add_argument("config", type=Path)
    options = parser.parse_args()
    config = RunConfig.load(options.config)
    arguments = config.arguments()
    if options.command == "plan":
        print(json.dumps({"argv": arguments, "runtime_validated": False}, indent=2))
        return
    # Runtime-only imports keep planning usable on CPU-only submitting hosts.
    native = importlib.import_module("miles.utils.arguments")
    sys.argv = [sys.argv[0], *arguments]
    args = native.parse_args()
    if args.train_backend != "olmo_core":
        raise RuntimeError("Expected the pinned MILES runtime with the olmo_core backend patch")
    if options.command == "validate":
        print("MILES arguments validated for OLMo-core")
        return
    if args.load:
        checkpoint = importlib.import_module("open_instruct.miles.checkpoint")
        _, manifest = checkpoint.resume_manifest(args.load)
        args.start_rollout_id = manifest["clock"]["next_rollout_id"]
    os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "olmo_sglang.models")
    driver = importlib.import_module("open_instruct.miles.driver")
    asyncio.run(driver.train(args))


if __name__ == "__main__":
    main()
