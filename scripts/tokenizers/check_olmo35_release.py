#!/usr/bin/env python
"""Check that a pinned Olmo 3.5 tokenizer states a concrete knowledge cutoff.

The cutoff is filled upstream in the tokenizer's chat template
(https://github.com/allenai/open-instruct/issues/2171). This script never edits a
template: it refuses any template that still contains the placeholder, reads the
Month YYYY cutoff from the rendered default system prompt, and records hashes.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from open_instruct import tokenizer_utils

PLACEHOLDER = "[CUTOFF_DATE]"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
CUTOFF_PATTERN = re.compile(rf"knowledge cutoff is ((?:{MONTHS}) [0-9]{{4}})\b")
SYSTEM_START, TURN_END, USER_START = "<|im_start|>system\n", "<|im_end|>\n", "<|im_start|>user\n"


def template_hash(template: str | dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()


def default_system(tokenizer) -> str:
    """Return the system turn the template injects when a chat has none."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=False
    )
    prefix, separator, _ = rendered.partition(USER_START)
    if tokenizer.bos_token and prefix.startswith(tokenizer.bos_token):
        prefix = prefix.removeprefix(tokenizer.bos_token)
    if not separator or not prefix.startswith(SYSTEM_START) or not prefix.endswith(TURN_END):
        raise ValueError("Expected an Olmo template with one default system turn")
    system = prefix[len(SYSTEM_START) : -len(TURN_END)]
    if "<|im_start|>" in system or "<|im_end|>" in system:
        raise ValueError("Default system turn must not contain additional turns")
    return system


def load_checked_tokenizer(tokenizer: str, revision: str):
    """Load a tokenizer at an immutable revision and verify its filled cutoff."""
    if not revision or (not Path(tokenizer).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("--revision must be a full 40-character Hub commit SHA for remote tokenizers")
    loaded = tokenizer_utils.load_tokenizer(tokenizer, revision=revision)
    template = loaded.chat_template
    templates = {"default": template} if isinstance(template, str) else template
    if not isinstance(templates, dict) or not templates or not all(isinstance(v, str) for v in templates.values()):
        raise ValueError("Expected a chat template string or nonempty dictionary of template strings")
    unfilled = sorted(name for name, value in templates.items() if PLACEHOLDER in value)
    if unfilled:
        raise ValueError(
            f"Chat template {', '.join(unfilled)} still contains {PLACEHOLDER}; fill it upstream "
            "(open-instruct#2171) and pass that revision"
        )
    system = default_system(loaded)
    cutoffs = CUTOFF_PATTERN.findall(system)
    if len(cutoffs) != 1 or cutoffs[0].endswith(" 0000"):
        raise ValueError("Default system prompt must state 'knowledge cutoff is <Month YYYY>' exactly once")
    receipt = {
        "source": str(Path(tokenizer).resolve()) if Path(tokenizer).is_dir() else tokenizer,
        "revision": revision,
        "cutoff_date": cutoffs[0],
        "default_system": system,
        "template_sha256": template_hash(template),
        "backend_sha256": hashlib.sha256(loaded.backend_tokenizer.to_str().encode()).hexdigest(),
    }
    return loaded, receipt


def check_tokenizer(tokenizer: str, revision: str, output: Path | None = None) -> dict:
    if output is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"Receipt output must be a new file: {output}")
    _, receipt = load_checked_tokenizer(tokenizer, revision)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Hub tokenizer ID or local tokenizer directory")
    parser.add_argument(
        "--revision", required=True, help="Full Hub commit SHA; provenance label for a local directory"
    )
    parser.add_argument("--output", type=Path, help="Optional new JSON file for the receipt")
    args = parser.parse_args()
    try:
        receipt = check_tokenizer(args.tokenizer, args.revision, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
