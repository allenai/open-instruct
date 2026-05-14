#!/usr/bin/env python3
"""Read a shell training script, override specific arguments, and print to stdout.

Usage:
    python scripts/update_command_args.py <script_path> [--key value ...]

Pipe the output to bash to execute with overrides without modifying the file:
    python scripts/update_command_args.py scripts/train/tulu3/grpo_fast_8b.sh \\
        --cluster ai2/jupiter --priority normal | uv run bash
"""

import re
import sys
from pathlib import Path


def parse_overrides(args: list[str]) -> dict[str, str | None]:
    """Parse --key value pairs from a flat list of strings."""
    overrides: dict[str, str | None] = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                overrides[key] = args[i + 1]
                i += 2
            else:
                overrides[key] = None
                i += 1
        else:
            i += 1
    return overrides


def update_script(script_text: str, overrides: dict[str, str | None]) -> str:
    for key, value in overrides.items():
        if value is None:
            continue
        pattern = rf"({re.escape(key)}\s+)\S+"
        new_text, count = re.subn(pattern, lambda m, v=value: m.group(1) + v, script_text)
        if count == 0:
            print(f"Warning: {key} not found in script, skipping.", file=sys.stderr)
        else:
            script_text = new_text
    return script_text


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <script_path> [--key value ...]", file=sys.stderr)
        sys.exit(1)

    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    overrides = parse_overrides(sys.argv[2:])
    modified = update_script(script_path.read_text(), overrides)
    print(modified, end="")


if __name__ == "__main__":
    main()
