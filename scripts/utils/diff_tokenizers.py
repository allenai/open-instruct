"""Compare two HuggingFace tokenizer repos file-by-file.

Downloads both repos to temp directories and diffs every file.
For tokenizer_config.json, extracts and pretty-prints the chat_template
separately so the actual difference is easy to spot.

Usage:
    python scripts/utils/diff_tokenizers.py allenai/olmo-3-tokenizer-instruct-dev allenai/olmo-3-tokenizer-instruct-release
    python scripts/utils/diff_tokenizers.py allenai/olmo-3.2-tokenizer-think-dev allenai/olmo-3.2-tokenizer-think-release
"""

import argparse
import difflib
import json
import os
import sys
import tempfile

from huggingface_hub import snapshot_download


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


def diff_chat_templates(path_a, path_b, repo_a, repo_b):
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

    print(f"\n  chat_template diff ({repo_a} vs {repo_b}):")
    # Pretty-print by splitting on template delimiters for readability
    lines_a = template_a.replace("-%}", "-%}\n").replace("%}", "%}\n").splitlines(keepends=True)
    lines_b = template_b.replace("-%}", "-%}\n").replace("%}", "%}\n").splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(lines_a, lines_b, fromfile=repo_a, tofile=repo_b, lineterm="")
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
        for line in difflib.unified_diff(a_str, b_str, fromfile=repo_a, tofile=repo_b):
            print(f"    {line}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Diff two HuggingFace tokenizer repos")
    parser.add_argument("repo_a", help="First tokenizer repo (e.g. allenai/olmo-3-tokenizer-instruct-dev)")
    parser.add_argument("repo_b", help="Second tokenizer repo (e.g. allenai/olmo-3-tokenizer-instruct-release)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading {args.repo_a}...")
        dir_a = snapshot_download(args.repo_a, local_dir=os.path.join(tmpdir, "a"))
        print(f"Downloading {args.repo_b}...")
        dir_b = snapshot_download(args.repo_b, local_dir=os.path.join(tmpdir, "b"))

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

        print(f"\nComparing {args.repo_a} vs {args.repo_b}")
        print("=" * 60)

        if only_a:
            print(f"\n  Only in {args.repo_a}: {', '.join(sorted(only_a))}")
        if only_b:
            print(f"\n  Only in {args.repo_b}: {', '.join(sorted(only_b))}")

        all_identical = True
        for name in common:
            path_a = os.path.join(dir_a, name)
            path_b = os.path.join(dir_b, name)

            if name == "tokenizer_config.json":
                identical = diff_files(path_a, path_b, name)
                if not identical:
                    diff_chat_templates(path_a, path_b, args.repo_a, args.repo_b)
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
