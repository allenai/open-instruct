"""Compare two HuggingFace tokenizer repos file-by-file.

Downloads both repos to temp directories and diffs every file.
For tokenizer_config.json, extracts and pretty-prints the chat_template
separately so the actual difference is easy to spot.

Usage:
    # Compare two different repos:
    python scripts/tokenizers/diff_tokenizers.py allenai/olmo-3-tokenizer-instruct-dev allenai/olmo-3-tokenizer-instruct-release

    # Compare the same repo at two different revisions:
    python scripts/tokenizers/diff_tokenizers.py allenai/olmo-3.2-tokenizer-think-dev --rev-a abc123 --rev-b main
"""

import argparse
import difflib
import json
import os
import re
import sys
import tempfile

from huggingface_hub import snapshot_download

# Only download tokenizer-related files, never model weights
TOKENIZER_PATTERNS = [
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
    "fix_tokens.py",
    "README.md",
]


def diff_files(path_a, path_b, name):
    """Diff two files, return True if identical."""
    with open(path_a) as f:
        lines_a = f.readlines()
    with open(path_b) as f:
        lines_b = f.readlines()

    if lines_a == lines_b:
        print(f"  {name}: identical")
        return True

    diff = list(
        difflib.unified_diff(lines_a, lines_b, fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="")
    )
    print(f"  {name}: DIFFERENT")
    for line in diff:
        print(f"    {line}")
    return False


def prettify_jinja(template):
    """Format a one-line Jinja chat template into readable indented lines."""
    # Split on Jinja block boundaries, keeping delimiters attached
    tokens = re.split(r"(\{%-?\s.*?-?%\}|\{\{-?\s.*?-?\}\})", template)
    lines = []
    indent = 0
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Dedent for closing/transition blocks before printing
        if re.match(r"\{%-?\s*(endif|endfor|else|elif)", token):
            indent = max(indent - 1, 0)
        lines.append("  " * indent + token)
        # Indent after opening blocks
        if re.match(r"\{%-?\s*(if|for|else|elif)\b", token) and "endif" not in token and "endfor" not in token:
            indent += 1
    return "\n".join(lines) + "\n"


def diff_chat_templates(path_a, path_b, label_a, label_b):
    """Extract and diff just the chat_template from tokenizer_config.json."""
    with open(path_a) as f:
        config_a = json.load(f)
    with open(path_b) as f:
        config_b = json.load(f)

    template_a = config_a.get("chat_template", "")
    template_b = config_b.get("chat_template", "")

    if template_a == template_b:
        print("\n  chat_template: identical")
        return True

    print(f"\n  chat_template diff ({label_a} vs {label_b}):")
    lines_a = prettify_jinja(template_a).splitlines(keepends=True)
    lines_b = prettify_jinja(template_b).splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm="")
    )
    for line in diff:
        print(f"    {line}")

    # Also show non-template diffs
    config_a_no_template = {k: v for k, v in config_a.items() if k != "chat_template"}
    config_b_no_template = {k: v for k, v in config_b.items() if k != "chat_template"}
    if config_a_no_template == config_b_no_template:
        print("\n  tokenizer_config.json (excluding chat_template): identical")
    else:
        print("\n  tokenizer_config.json (excluding chat_template): DIFFERENT")
        a_str = json.dumps(config_a_no_template, indent=2).splitlines(keepends=True)
        b_str = json.dumps(config_b_no_template, indent=2).splitlines(keepends=True)
        for line in difflib.unified_diff(a_str, b_str, fromfile=label_a, tofile=label_b):
            print(f"    {line}")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Diff two HuggingFace tokenizer repos (or the same repo at two revisions)"
    )
    parser.add_argument("repo_a", help="First tokenizer repo (e.g. allenai/olmo-3-tokenizer-instruct-dev)")
    parser.add_argument(
        "repo_b",
        nargs="?",
        default=None,
        help="Second tokenizer repo. If omitted, compares repo_a at two revisions (use --rev-a and --rev-b).",
    )
    parser.add_argument("--rev-a", default=None, help="Git revision (commit SHA, branch, tag) for repo_a")
    parser.add_argument("--rev-b", default=None, help="Git revision (commit SHA, branch, tag) for repo_b")
    args = parser.parse_args()

    # If no repo_b, compare the same repo at two revisions
    if args.repo_b is None:
        args.repo_b = args.repo_a
        if args.rev_a is None or args.rev_b is None:
            parser.error("When comparing a single repo at two revisions, both --rev-a and --rev-b are required.")

    # Build labels for display
    label_a = args.repo_a + (f"@{args.rev_a}" if args.rev_a else "")
    label_b = args.repo_b + (f"@{args.rev_b}" if args.rev_b else "")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading {label_a}...")
        dl_kwargs_a = {"repo_id": args.repo_a, "local_dir": os.path.join(tmpdir, "a"), "allow_patterns": TOKENIZER_PATTERNS}
        if args.rev_a:
            dl_kwargs_a["revision"] = args.rev_a
        dir_a = snapshot_download(**dl_kwargs_a)

        print(f"Downloading {label_b}...")
        dl_kwargs_b = {"repo_id": args.repo_b, "local_dir": os.path.join(tmpdir, "b"), "allow_patterns": TOKENIZER_PATTERNS}
        if args.rev_b:
            dl_kwargs_b["revision"] = args.rev_b
        dir_b = snapshot_download(**dl_kwargs_b)

        # Collect all files (skip hidden dirs like .cache)
        files_a = {
            f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f)) and not f.startswith(".")
        }
        files_b = {
            f for f in os.listdir(dir_b) if os.path.isfile(os.path.join(dir_b, f)) and not f.startswith(".")
        }

        only_a = files_a - files_b
        only_b = files_b - files_a
        common = sorted(files_a & files_b)

        print(f"\nComparing {label_a} vs {label_b}")
        print("=" * 60)

        if only_a:
            print(f"\n  Only in {label_a}: {', '.join(sorted(only_a))}")
        if only_b:
            print(f"\n  Only in {label_b}: {', '.join(sorted(only_b))}")

        all_identical = True
        for name in common:
            path_a = os.path.join(dir_a, name)
            path_b = os.path.join(dir_b, name)

            if name == "tokenizer_config.json":
                # Skip raw diff (unreadable for chat_template), show pretty version
                with open(path_a) as f:
                    raw_a = f.read()
                with open(path_b) as f:
                    raw_b = f.read()
                if raw_a == raw_b:
                    print(f"  {name}: identical")
                else:
                    print(f"  {name}: DIFFERENT (see chat_template diff below)")
                    diff_chat_templates(path_a, path_b, label_a, label_b)
                    all_identical = False
            elif name == "chat_template.jinja":
                # Pretty-print jinja diff
                with open(path_a) as f:
                    raw_a = f.read()
                with open(path_b) as f:
                    raw_b = f.read()
                if raw_a == raw_b:
                    print(f"  {name}: identical")
                else:
                    print(f"  {name}: DIFFERENT")
                    lines_a = prettify_jinja(raw_a).splitlines(keepends=True)
                    lines_b = prettify_jinja(raw_b).splitlines(keepends=True)
                    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm=""))
                    for line in diff:
                        print(f"    {line}")
                    all_identical = False
            else:
                if not diff_files(path_a, path_b, name):
                    all_identical = False

        print("\n" + "=" * 60)
        if all_identical and not only_a and not only_b:
            print("Result: tokenizers are IDENTICAL")
        else:
            print("Result: tokenizers DIFFER (see above)")

    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
