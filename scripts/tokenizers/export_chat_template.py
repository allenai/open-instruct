#!/usr/bin/env python
"""Save a chat template defined in open_instruct.dataset_transformation to a Jinja file."""

import argparse
import sys
from pathlib import Path

from open_instruct.dataset_transformation import CHAT_TEMPLATES

# Example
# uv run python scripts/export_chat_template.py olmo_thinker_remove_intermediate_thinking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "template", help="Name of the chat template as defined in open_instruct.dataset_transformation.CHAT_TEMPLATES."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the template (defaults to TEMPLATE_NAME.jinja in the current directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing file; otherwise the script exits with an error.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available template names and exit (Ignores other flags)."
    )
    return parser.parse_args()


def list_templates() -> None:
    print("Available chat templates:")
    for name in sorted(CHAT_TEMPLATES.keys()):
        print(f" - {name}")


def main() -> None:
    args = parse_args()

    if args.list:
        list_templates()
        return

    if args.template not in CHAT_TEMPLATES:
        print(f"Unknown template '{args.template}'. Use --list for options.", file=sys.stderr)
        raise SystemExit(1)

    template_str = CHAT_TEMPLATES[args.template]
    # ensure POSIX newlines
    template_str = template_str.replace("\r\n", "\n")

    output_path = args.output or Path(f"{args.template}.jinja")
    if output_path.exists() and not args.overwrite:
        print(f"{output_path} already exists. Use --overwrite to replace it.", file=sys.stderr)
        raise SystemExit(1)

    output_path.write_text(template_str, encoding="utf-8")
    print(f"Wrote {args.template} template to {output_path}")


if __name__ == "__main__":
    main()
