"""Capture the installed upstream backend parsers inside the application image."""

import argparse
import json
import os
import sys
from pathlib import Path

from megatron.training import arguments as megatron_arguments
from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.backends.torchtitan_utils import arguments as titan_arguments
from miles.utils import arguments as miles_arguments

from open_instruct.miles.configuration import options


class CapturedParser(Exception):
    def __init__(self, parser):
        self.parser = parser


def parser_for(backend):
    provider = miles_arguments.get_miles_extra_args_provider()
    if backend == "fsdp":
        return fsdp_arguments.build_fsdp_parser(provider)
    if backend == "torchtitan":
        return titan_arguments.build_torchtitan_parser(provider)

    def capture(parser):
        raise CapturedParser(provider(parser))

    # Use Megatron's public parser entrypoint, stopping before required-argument
    # checks or side effects. This also works when its internal builder changes.
    try:
        megatron_arguments.parse_args(extra_args_provider=capture)
    except CapturedParser as captured:
        return captured.parser
    raise RuntimeError("Megatron did not expose its argument parser")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("destination", type=Path)
    args = cli.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path("/opt/miles-upstream/runtime.lock.json").read_text())
    for backend in ("fsdp", "megatron", "torchtitan"):
        # Upstream probes argv while registering dynamic rollout arguments.
        sys.argv = ["snapshot", "--train-backend", backend]
        parser = parser_for(backend)
        schema = {
            "schema_version": 1,
            "source": lock["miles"],
            "backend": backend,
            "build_id": os.environ["MILES_UPSTREAM_BUILD_ID"],
            "options": options.compact_options(options.describe_parser(parser)),
        }
        (args.destination / f"{backend}.json").write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main()
