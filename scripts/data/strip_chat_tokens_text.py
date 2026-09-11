"""Make a "chat-tokens-removed" copy of an Olmo 3.5 plain-text dataset.

Takes a dataset produced by render_olmo35_text.py (columns id, dataset_source, text, n_tokens) and
removes the chat-format markup from `text`, keeping everything else verbatim:

* role headers  `<|im_start|>system` / `user` / `assistant` / `environment` (with the newline that
  follows them) and the turn terminator `<|im_end|>`;
* the end-of-document token `<|endoftext|>`;
* the reasoning delimiters `<think>` and `</think>` (the reasoning text itself stays).

The tool-calling markup (`<tools>`, `<tool_call>`, `<function=…>`, `<parameter=…>`,
`<tool_response>`) is kept. `n_tokens` is recomputed under the same tokenizer. Rows keep their order
and ids. The result is refused if any `<|…|>` token or think tag survives.

    python scripts/data/strip_chat_tokens_text.py --source allenai/simfc-thinking-qwen35-olmo35-text \\
        --out-dir /tmp/simfc-chat-tokens-removed \\
        --push-to allenai/simfc-thinking-qwen35-olmo35-text-chat-tokens-removed --private
"""

import argparse
import glob
import json
import multiprocessing
import os
import pathlib
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
SCHEMA = pa.schema(
    [("id", pa.string()), ("dataset_source", pa.string()), ("text", pa.string()), ("n_tokens", pa.int32())]
)
BATCH_ROWS = 2000
ROLE_HEADER = re.compile(r"<\|im_start\|>[a-z_]+\n?")
LITERALS = ("<|im_end|>", "<|endoftext|>", "<think>", "</think>")
LEFTOVER = re.compile(r"<\|[a-z_]+\|>|</?think>")

_tokenizer = None


def _get_tokenizer(name: str, revision: str):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    return _tokenizer


def strip_chat_tokens(text: str) -> str:
    text = ROLE_HEADER.sub("", text)
    for literal in LITERALS:
        text = text.replace(literal, "")
    return text.strip("\n")


def process_shard(args: tuple[str, str, str, str, int | None]) -> tuple[str, int, int, int, int]:
    path, out_dir, tokenizer_name, tokenizer_revision, limit = args
    tokenizer = _get_tokenizer(tokenizer_name, tokenizer_revision)
    dst = os.path.join(out_dir, "data", os.path.basename(path))
    n_rows = n_tokens_total = n_tokens_max = n_tokens_before = 0
    with pq.ParquetWriter(dst, SCHEMA, compression="zstd") as writer:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
            out: dict[str, list] = {"id": [], "dataset_source": [], "text": [], "n_tokens": []}
            for row in batch.to_pylist():
                if limit is not None and n_rows >= limit:
                    break
                text = strip_chat_tokens(row["text"])
                if LEFTOVER.search(text):
                    raise ValueError(f"chat markup survived in row {row['id']}: {LEFTOVER.search(text).group(0)!r}")
                n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                out["id"].append(row["id"])
                out["dataset_source"].append(row["dataset_source"])
                out["text"].append(text)
                out["n_tokens"].append(n)
                n_rows += 1
                n_tokens_total += n
                n_tokens_before += row["n_tokens"]
                n_tokens_max = max(n_tokens_max, n)
            if out["id"]:
                writer.write_table(pa.table(out, schema=SCHEMA))
            if limit is not None and n_rows >= limit:
                break
    return dst, n_rows, n_tokens_total, n_tokens_max, n_tokens_before


def write_readme(out_dir: pathlib.Path, prov: dict[str, Any]) -> None:
    size = sum(p.stat().st_size for p in (out_dir / "data").glob("*.parquet"))
    source = prov["source_repo"]
    (out_dir / "README.md").write_text(
        f"""---
license: apache-2.0
language: [en]
dataset_info:
  features:
  - name: id
    dtype: string
  - name: dataset_source
    dtype: string
  - name: text
    dtype: string
  - name: n_tokens
    dtype: int32
  splits:
  - name: train
    num_examples: {prov["rows"]}
  download_size: {size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# {source.split("/")[-1]}, chat tokens removed

[{source}](https://huggingface.co/datasets/{source}) with the Olmo 3.5 chat-format markup stripped
out of `text`; everything else is verbatim, in the same order, with the same `id`s.

| | |
|---|---|
| source | `{source}` @ `{prov["source_revision"]}` |
| tokenizer for `n_tokens` | `{prov["tokenizer"]}` @ `{prov["tokenizer_revision"]}` |
| rows | {prov["rows"]:,} |
| tokens | {prov["tokens"]:,} (source: {prov["tokens_before"]:,}), max {prov["max_tokens"]:,} per row |

## What was removed

- the role headers `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` and
  `<|im_start|>environment`, together with the newline that follows each, and the turn terminator
  `<|im_end|>`;
- the end-of-document token `<|endoftext|>`;
- the reasoning delimiters `<think>` and `</think>`; the reasoning text between them is kept.

## What was kept

The tool-calling markup the model is meant to learn: the `<tools>` block in the system prompt,
`<tool_call>` / `<function=…>` / `<parameter=…>` blocks in assistant turns and `<tool_response>`
wrappers around tool results. Turns therefore follow one another separated by a newline, with no
role labels. `n_tokens` was recomputed after the removal.

Produced by `scripts/data/strip_chat_tokens_text.py` in [allenai/open-instruct](https://github.com/allenai/open-instruct).
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="HF dataset produced by render_olmo35_text.py")
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--num-proc", type=int, default=None, help="worker processes (default: one per shard)")
    parser.add_argument("--limit", type=int, default=None, help="rows per shard, for a quick check")
    parser.add_argument("--push-to", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    source_sha = args.source_revision or api.dataset_info(args.source).sha
    tokenizer_sha = api.model_info(args.tokenizer).sha
    logger.info(f"source {args.source}@{source_sha}, tokenizer {args.tokenizer}@{tokenizer_sha}")
    src = snapshot_download(args.source, repo_type="dataset", revision=source_sha, allow_patterns=["data/*.parquet"])
    files = sorted(glob.glob(os.path.join(src, "data", "*.parquet")))
    if not files:
        raise SystemExit(f"no data/*.parquet in {args.source}@{source_sha}")
    if args.limit is not None:
        files = files[:1]
    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    jobs = [(f, str(out_dir), args.tokenizer, tokenizer_sha, args.limit) for f in files]
    with multiprocessing.get_context("spawn").Pool(min(args.num_proc or len(files), len(files))) as pool:
        results = pool.map(process_shard, jobs)
    prov = {
        "source_repo": args.source,
        "source_revision": source_sha,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": tokenizer_sha,
        "rows": sum(r[1] for r in results),
        "tokens": sum(r[2] for r in results),
        "max_tokens": max(r[3] for r in results),
        "tokens_before": sum(r[4] for r in results),
        "removed": ["<|im_start|>{role}\\n", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"],
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    write_readme(out_dir, prov)
    logger.info(
        f"{prov['rows']:,} rows, {prov['tokens']:,} tokens (source {prov['tokens_before']:,}), max {prov['max_tokens']:,}"
    )
    if args.push_to:
        if args.limit is not None:
            raise SystemExit("refusing to push a --limit sample")
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        info = api.upload_folder(
            repo_id=args.push_to,
            repo_type="dataset",
            folder_path=str(out_dir),
            commit_message=f"{args.source}@{source_sha[:8]} with chat tokens removed",
        )
        logger.info(f"pushed to {info.commit_url}")


if __name__ == "__main__":
    main()
