#!/usr/bin/env python
"""Build hand-coded identity SFT rows from a tokenizer whose cutoff is filled upstream.

The output directory contains identity.jsonl and manifest.json. Each seed gets
an implicit-system and an explicit-default-system variant with identical renders.
"""

import argparse
import hashlib
import json
import re
import tempfile
from importlib import util
from pathlib import Path

SEEDS = Path(__file__).with_name("olmo35_identity_seed.json")
SOURCE = "olmo35-identity-hand-coded"
# Canonical sorted {id, user} records from the 58 audited identity rows, not
# from the new responses. Changing a prompt must be reconciled with that audit.
# Source: k3-audit/verify-ho2/u05-identity-stale--mass-and-recall/03_identity_rows.parquet
# File SHA256: 32ef847c6b0a18c027ccc9fe18e4d9cb0facd2cb20bbea0b32c98515a93cf60d
AUDITED_PROMPTS_SHA256 = "94948894630e308362e49c17b81ed4413c90d1fb95dc50c1203f1cf25553b991"
CHECKER = Path(__file__).resolve().parents[2] / "tokenizers/check_olmo35_release.py"


def _load_checker():
    spec = util.spec_from_file_location("check_olmo35_release", CHECKER)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_tokenizer = _load_checker()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_identity(
    tokenizer_name: str, revision: str, output: Path, *, context_length: int, context_source: str
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be a fresh directory: {output}")
    if type(context_length) is not int or context_length <= 0 or not context_source.strip():
        raise ValueError("A positive context length and its verification source are required")
    tokenizer, receipt = release_tokenizer.load_checked_tokenizer(tokenizer_name, revision)
    system = receipt["default_system"]
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
    parser.add_argument("--tokenizer", required=True, help="Hub tokenizer ID or local tokenizer directory")
    parser.add_argument(
        "--revision", required=True, help="Full Hub commit SHA with the cutoff filled (open-instruct#2171)"
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
            args.tokenizer,
            args.revision,
            args.output,
            context_length=args.context_length,
            context_source=args.context_source,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {manifest['rows']} identity rows to {args.output / 'identity.jsonl'}")


if __name__ == "__main__":
    main()
