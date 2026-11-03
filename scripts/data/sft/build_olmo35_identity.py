#!/usr/bin/env python
"""Build hand-coded identity SFT rows only after preparing a dated release tokenizer.

The output directory contains identity.jsonl and manifest.json. Each seed gets
an implicit-system and an explicit-default-system variant with identical renders.
"""

import argparse
import hashlib
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

from open_instruct import tokenizer_utils

SEEDS = Path(__file__).with_name("olmo35_identity_seed.json")
SOURCE = "olmo35-identity-hand-coded"
# Canonical sorted {id, user} records from the 58 audited identity rows, not
# from the new responses. Changing a prompt must be reconciled with that audit.
# Source: k3-audit/verify-ho2/u05-identity-stale--mass-and-recall/03_identity_rows.parquet
# File SHA256: 32ef847c6b0a18c027ccc9fe18e4d9cb0facd2cb20bbea0b32c98515a93cf60d
AUDITED_PROMPTS_SHA256 = "94948894630e308362e49c17b81ed4413c90d1fb95dc50c1203f1cf25553b991"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_prepared_tokenizer(directory: Path):
    receipt_file = directory / "release_tokenizer.json"
    if not receipt_file.is_file():
        raise ValueError("A prepared tokenizer with release_tokenizer.json is required; cutoff remains pending")
    receipt = json.loads(receipt_file.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("Invalid prepared-tokenizer receipt")
    for key in ("source", "revision", "cutoff_date"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ValueError(f"Invalid prepared-tokenizer receipt field: {key}")
    for key in ("source_template_sha256", "template_sha256", "backend_sha256"):
        if not isinstance(receipt.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[key]):
            raise ValueError(f"Invalid prepared-tokenizer receipt field: {key}")
    try:
        date = datetime.strptime(receipt["cutoff_date"], "%B %Y")
    except ValueError as error:
        raise ValueError("Prepared tokenizer must have an explicit Month YYYY cutoff") from error
    if date.strftime("%B %Y") != receipt["cutoff_date"]:
        raise ValueError("Prepared tokenizer must have an explicit Month YYYY cutoff")
    files = {
        path.relative_to(directory).as_posix(): _sha256(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != receipt_file
    }
    if not files or files != receipt.get("files_sha256"):
        raise ValueError("Prepared tokenizer files do not match its receipt")
    tokenizer = tokenizer_utils.load_tokenizer(directory)
    template = tokenizer.chat_template
    if _sha256(json.dumps(template, sort_keys=True).encode()) != receipt["template_sha256"]:
        raise ValueError("Prepared tokenizer template hash does not match its receipt")
    if _sha256(tokenizer.backend_tokenizer.to_str().encode()) != receipt["backend_sha256"]:
        raise ValueError("Prepared tokenizer backend hash does not match its receipt")
    if "[CUTOFF_DATE]" in json.dumps(template):
        raise ValueError("Prepared tokenizer still contains [CUTOFF_DATE]")
    return tokenizer, receipt


def _default_system(tokenizer, cutoff_date: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=False
    )
    prefix, separator, _ = rendered.partition("<|im_start|>user\n")
    if tokenizer.bos_token and prefix.startswith(tokenizer.bos_token):
        prefix = prefix.removeprefix(tokenizer.bos_token)
    start, end = "<|im_start|>system\n", "<|im_end|>\n"
    if not separator or not prefix.startswith(start) or not prefix.endswith(end):
        raise ValueError("Expected an Olmo template with one default system turn")
    system = prefix[len(start) : -len(end)]
    if "<|im_start|>" in system or "<|im_end|>" in system or cutoff_date not in system:
        raise ValueError("Default system must contain the prepared cutoff and no additional turns")
    return system


def build_identity(tokenizer_dir: Path, output: Path, *, context_length: int, context_source: str) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a fresh directory: {output}")
    if type(context_length) is not int or context_length <= 0 or not context_source.strip():
        raise ValueError("A positive context length and its verification source are required")
    tokenizer, receipt = _load_prepared_tokenizer(tokenizer_dir)
    system = _default_system(tokenizer, receipt["cutoff_date"])
    seeds = json.loads(SEEDS.read_text())
    if len(seeds) != 58 or len({seed["id"] for seed in seeds}) != 58:
        raise ValueError("Expected exactly 58 distinct identity seeds")
    prompts = sorted(({"id": seed["id"], "user": seed["user"]} for seed in seeds), key=lambda row: row["id"])
    if _sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False).encode()) != AUDITED_PROMPTS_SHA256:
        raise ValueError("Identity IDs or prompts differ from the 58 audited rows")
    rows = []
    for seed in seeds:
        answer = (
            seed["assistant"]
            .replace("{cutoff_date}", receipt["cutoff_date"])
            .replace("{context_length}", f"{context_length:,}")
        )
        if re.search(r"\{\w+\}|\[CUTOFF_DATE\]|</?think>", answer):
            raise ValueError(f"Unresolved placeholder or reasoning trace in {seed['id']}")
        implicit = [{"role": "user", "content": seed["user"]}, {"role": "assistant", "content": answer}]
        explicit = [{"role": "system", "content": system}, *implicit]
        for generation_prompt in (False, True):
            for tokenize in (False, True):
                kwargs = {"tokenize": tokenize, "add_generation_prompt": generation_prompt}
                if tokenizer.apply_chat_template(implicit, **kwargs) != tokenizer.apply_chat_template(
                    explicit, **kwargs
                ):
                    raise ValueError(f"Explicit and implicit default systems render differently for {seed['id']}")
        for variant, messages in (("implicit", implicit), ("explicit", explicit)):
            rows.append(
                {
                    "id": f"olmo35-identity-{seed['id']}-{variant}",
                    "parent_id": seed["id"],
                    "source": SOURCE,
                    "messages": messages,
                }
            )
    manifest = {
        "source": SOURCE,
        "rows": len(rows),
        "seed_sha256": _sha256(SEEDS.read_bytes()),
        "audited_prompts_sha256": AUDITED_PROMPTS_SHA256,
        "context_length": context_length,
        "context_source": context_source,
        "tokenizer": str(tokenizer_dir.resolve()),
        "tokenizer_receipt": receipt,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "identity"
        staged.mkdir()
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (staged / "identity.jsonl").write_text(data)
        manifest["data_sha256"] = _sha256(data.encode())
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output must be a fresh directory: {output}")
        staged.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Prepared tokenizer directory with cutoff receipt"
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument(
        "--context-length", type=int, required=True, help="Verified context length of the release model"
    )
    parser.add_argument(
        "--context-source", required=True, help="Pinned model card/config reference verifying that length"
    )
    args = parser.parse_args()
    try:
        manifest = build_identity(
            args.tokenizer, args.output, context_length=args.context_length, context_source=args.context_source
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {manifest['rows']} identity rows to {args.output / 'identity.jsonl'}")


if __name__ == "__main__":
    main()
