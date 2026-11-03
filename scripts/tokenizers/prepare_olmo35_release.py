#!/usr/bin/env python
"""Prepare a concrete Olmo 3.5 tokenizer for SFT, HF export and serving.

Reuse the output directory as the training tokenizer and export --tokenizer-name.
Remote inputs require a full Hub commit SHA; no cutoff date is assumed.
"""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from open_instruct import tokenizer_utils

PLACEHOLDER = "[CUTOFF_DATE]"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def template_hash(template: str | dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()


def prepare_tokenizer(tokenizer: str, revision: str, cutoff_date: str, output: Path) -> dict:
    if not re.fullmatch(r"(" + "|".join(MONTHS) + r") [0-9]{4}", cutoff_date) or cutoff_date.endswith(" 0000"):
        raise ValueError("--cutoff-date must be a real month and year in Month YYYY form, e.g. January 2025")
    if not revision or (not Path(tokenizer).is_dir() and not re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
        raise ValueError("--revision must be a full 40-character Hub commit SHA for remote tokenizers")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a fresh directory: {output}")

    loaded = tokenizer_utils.load_tokenizer(tokenizer, revision=revision)
    original = loaded.chat_template
    if isinstance(original, str):
        templates = {"default": original}
    elif isinstance(original, dict) and original and all(isinstance(value, str) for value in original.values()):
        templates = original
    else:
        raise ValueError("Expected a chat template string or nonempty dictionary of template strings")
    if not any(PLACEHOLDER in value for value in templates.values()):
        raise ValueError(f"Source chat template does not contain {PLACEHOLDER}")
    replaced = {name: value.replace(PLACEHOLDER, cutoff_date) for name, value in templates.items()}
    loaded.chat_template = replaced["default"] if isinstance(original, str) else replaced
    backend = loaded.backend_tokenizer.to_str()

    receipt = {
        "source": str(Path(tokenizer).resolve()) if Path(tokenizer).is_dir() else tokenizer,
        "revision": revision,
        "cutoff_date": cutoff_date,
        "source_template_sha256": template_hash(original),
        "template_sha256": template_hash(loaded.chat_template),
        "backend_sha256": hashlib.sha256(backend.encode()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "tokenizer"
        loaded.save_pretrained(staged)
        # Transformers writes standalone Jinja; also synchronize the config for
        # consumers that read it directly, using HF's serialized named-template format.
        config_file = staged / "tokenizer_config.json"
        config = json.loads(config_file.read_text())
        config["chat_template"] = (
            loaded.chat_template
            if isinstance(original, str)
            else [{"name": name, "template": value} for name, value in replaced.items()]
        )
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        restored = tokenizer_utils.load_tokenizer(staged)
        if restored.chat_template != loaded.chat_template or restored.backend_tokenizer.to_str() != backend:
            raise ValueError("Saved tokenizer changed its template or serialized backend")
        for name in templates:
            for explicit_system in (False, True):
                messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello"}]
                if explicit_system:
                    messages.insert(0, {"role": "system", "content": "Reply briefly."})
                for generation_prompt in (False, True):
                    kwargs = {"chat_template": name, "add_generation_prompt": generation_prompt}
                    # Named lookup only applies when the tokenizer has a dictionary.
                    if isinstance(original, str):
                        kwargs.pop("chat_template")
                    for tokenize in (False, True):
                        if loaded.apply_chat_template(
                            messages, tokenize=tokenize, **kwargs
                        ) != restored.apply_chat_template(messages, tokenize=tokenize, **kwargs):
                            raise ValueError("Saved tokenizer changed rendered chats or token IDs")
        receipt["files_sha256"] = {
            path.relative_to(staged).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        }
        (staged / "release_tokenizer.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output must be a fresh directory: {output}")
        staged.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Hub tokenizer ID or local tokenizer directory")
    parser.add_argument(
        "--revision", required=True, help="Full Hub commit SHA; provenance label for a local directory"
    )
    parser.add_argument("--cutoff-date", required=True, help="Knowledge cutoff in Month YYYY form")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    args = parser.parse_args()
    try:
        prepare_tokenizer(args.tokenizer, args.revision, args.cutoff_date, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared tokenizer: {args.output}")


if __name__ == "__main__":
    main()
