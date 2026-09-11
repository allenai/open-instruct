"""Write the CPU planning schema using the installed, pinned MILES/SGLang runtime.

Usage inside the runtime: python scripts/miles/snapshot_options.py /tmp/options.json
Review the diff and copy to open_instruct/miles/options.json when updating pins.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.utils import arguments

from open_instruct.miles.options import describe_parser


def snapshot():
    # SGLang probes parser construction with incomplete argv; suppress its handled usage errors.
    with contextlib.redirect_stderr(io.StringIO()):
        parser = fsdp_arguments.build_fsdp_parser(arguments.get_miles_extra_args_provider())
    parser.add_argument("--olmo-core-config", required=True)
    lock = json.loads(Path("runtime/miles/runtime.lock.json").read_text())
    return {"sources": lock["sources"], "options": describe_parser(parser)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = snapshot()
    # One action per line keeps the large upstream serving option surface reviewable.
    lines = ",\n".join("    " + json.dumps(record, sort_keys=True) for record in data["options"])
    args.output.write_text(
        '{\n  "sources": ' + json.dumps(data["sources"], sort_keys=True) + ',\n  "options": [\n' + lines + "\n  ]\n}\n"
    )
    print(f"Wrote {len(data['options'])} option definitions to {args.output}")


if __name__ == "__main__":
    main()
