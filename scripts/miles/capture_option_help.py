"""Capture native help in the pinned image; does not change the parser contract.

python -m scripts.miles.capture_option_help --image IMAGE_ID
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

from miles.backends.fsdp_utils import arguments as fsdp_arguments
from miles.utils import arguments

from open_instruct.miles import options


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--image", required=True, help="Immutable image identity used for this capture")
    args = cli.parse_args()
    root = Path(__file__).resolve().parents[2]
    schema = root / "open_instruct/miles/options.json"
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        parser = fsdp_arguments.build_fsdp_parser(arguments.get_miles_extra_args_provider())
        parser.add_argument("--olmo-core-config", required=True)
    if options.describe_parser(parser) != json.loads(schema.read_text())["options"]:
        raise ValueError("Installed parser differs from the checked-in schema; update/review the schema first")
    records = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        default = action.default
        try:
            json.dumps(default, allow_nan=False)
        except (TypeError, ValueError):
            default = "runtime object; inspect installed parser"
        records.append(
            dict(
                dest=action.dest,
                flags=action.option_strings,
                help=action.help if action.help != argparse.SUPPRESS else None,
                default=default,
            )
        )
    result = dict(image=args.image, schema_sha256=hashlib.sha256(schema.read_bytes()).hexdigest(), options=records)
    target = root / "docs/miles/native-help.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Captured {len(records)} actions in {target}")


if __name__ == "__main__":
    main()
