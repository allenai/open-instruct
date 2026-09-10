"""Render a messages-format SFT dataset with the Olmo 3.5 chat template into plain-text rows.

For midtraining, which consumes text rather than messages: every conversation becomes one `text`
string, exactly what the SFT tokenizer would see with `add_generation_prompt=False` (system prompt
with the `<tools>` block, `<think>` reasoning, XML tool calls, `<tool_response>` wrappers,
`<|im_end|>` between turns and `<|endoftext|>` at the end; no BOS). `id` and the source's
provenance column (`dataset_source` or `source`) are carried over; `n_tokens` is the length under
the same tokenizer, for packing and budgeting.

Works for any dataset in this repo's SFT format: `messages` with `role` / `content` /
`reasoning_content` / `tool_calls[].function.{name,arguments}` (arguments a JSON string) /
`tool_call_id`, plus an optional top-level `tools` JSON string. Shards are streamed in row-group
batches, so memory stays flat regardless of shard size.

    python scripts/data/render_olmo35_text.py --source allenai/simfc-thinking-qwen35 \\
        --out-dir /weka/oe-adapt-default/<user>/simfc-olmo35-text \\
        --push-to allenai/simfc-thinking-qwen35-olmo35-text --private
    python scripts/data/render_olmo35_text.py --source allenai/nemotron-sft-agentic-v2-tool-calling-oi \\
        --out-dir /weka/oe-adapt-default/<user>/nemotron-olmo35-text \\
        --push-to allenai/nemotron-sft-agentic-v2-tool-calling-oi-olmo35-text --private
"""

import argparse
import glob
import json
import multiprocessing
import os
import pathlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

DEFAULT_SOURCE = "allenai/simfc-thinking-qwen35"
DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
SOURCE_COLUMNS = ("dataset_source", "source")
SCHEMA = pa.schema(
    [("id", pa.string()), ("dataset_source", pa.string()), ("text", pa.string()), ("n_tokens", pa.int32())]
)
BATCH_ROWS = 2000

_tokenizer = None


def _get_tokenizer(name: str, revision: str):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    return _tokenizer


def _to_chat(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list | None]:
    messages = []
    for m in row["messages"]:
        m = {k: v for k, v in m.items() if v is not None}
        for tc in m.get("tool_calls") or []:
            arguments = tc["function"]["arguments"]
            if isinstance(arguments, str):
                tc["function"]["arguments"] = json.loads(arguments)
        messages.append(m)
    tools = json.loads(row["tools"]) if row.get("tools") else None
    return messages, tools


def render_shard(args: tuple[str, str, str, str, str, int | None]) -> tuple[str, int, int, int]:
    path, out_dir, tokenizer_name, tokenizer_revision, source_column, limit = args
    tokenizer = _get_tokenizer(tokenizer_name, tokenizer_revision)
    dst = os.path.join(out_dir, "data", os.path.basename(path))
    n_rows = n_tokens_total = n_tokens_max = 0
    columns = ["id", source_column, "messages", "tools"]
    with pq.ParquetWriter(dst, SCHEMA, compression="zstd") as writer:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
            out: dict[str, list] = {"id": [], "dataset_source": [], "text": [], "n_tokens": []}
            for row in batch.to_pylist():
                if limit is not None and n_rows >= limit:
                    break
                messages, tools = _to_chat(row)
                text = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, add_generation_prompt=False
                )
                n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                out["id"].append(row["id"])
                out["dataset_source"].append(row[source_column])
                out["text"].append(text)
                out["n_tokens"].append(n)
                n_rows += 1
                n_tokens_total += n
                n_tokens_max = max(n_tokens_max, n)
            if out["id"]:
                writer.write_table(pa.table(out, schema=SCHEMA))
            if limit is not None and n_rows >= limit:
                break
    return dst, n_rows, n_tokens_total, n_tokens_max


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

# {source.split("/")[-1]}, rendered with the Olmo 3.5 chat template

Plain-text version of [{source}](https://huggingface.co/datasets/{source}) for
midtraining: every conversation is rendered into one `text` string with the Olmo 3.5 chat template,
exactly as the tokenizer renders it for SFT with `add_generation_prompt=False`.

| | |
|---|---|
| source | `{source}` @ `{prov["source_revision"]}` |
| template | `{prov["tokenizer"]}` @ `{prov["tokenizer_revision"]}` |
| rows | {prov["rows"]:,} (every source row; nothing filtered) |
| tokens | {prov["tokens"]:,} under that tokenizer, max {prov["max_tokens"]:,} per row |

## What a row looks like

`<|im_start|>system` with the tool schemas in a `<tools>` block and the XML tool-calling instructions,
`<|im_start|>user` turns, `<|im_start|>assistant` turns as `<think>reasoning</think>` followed by the
answer and/or `<tool_call><function=...><parameter=...>` blocks, tool results as
`<|im_start|>environment` turns wrapping `<tool_response>`. Non-final assistant turns close with
`<|im_end|>`, the final one with `<|endoftext|>`. No BOS token is prepended.

The source's `reasoning_content` field becomes the `<think>` block; `tool_calls[].function.arguments`
(a JSON string in the source) is parsed to a mapping before rendering, as chat-template consumers
expect. `n_tokens` is the length under the same tokenizer. `id` and `dataset_source` (the source's
`{prov["source_column"]}` column) are carried over from the source for provenance.

Produced by `scripts/data/render_olmo35_text.py` in [allenai/open-instruct](https://github.com/allenai/open-instruct).
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="HF dataset repo in this repo's SFT messages format")
    parser.add_argument("--source-revision", default=None, help="commit to render (default: current main)")
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
    names = pq.ParquetFile(files[0]).schema_arrow.names
    source_column = next((c for c in SOURCE_COLUMNS if c in names), None)
    if source_column is None:
        raise SystemExit(f"{args.source} has none of {SOURCE_COLUMNS}; columns are {names}")
    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    num_proc = args.num_proc or len(files)
    jobs = [(f, str(out_dir), args.tokenizer, tokenizer_sha, source_column, args.limit) for f in files]
    with multiprocessing.get_context("spawn").Pool(min(num_proc, len(files))) as pool:
        results = pool.map(render_shard, jobs)
    prov = {
        "source_repo": args.source,
        "source_revision": source_sha,
        "source_column": source_column,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": tokenizer_sha,
        "rows": sum(r[1] for r in results),
        "tokens": sum(r[2] for r in results),
        "max_tokens": max(r[3] for r in results),
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1))
    write_readme(out_dir, prov)
    logger.info(f"rendered {prov['rows']:,} rows, {prov['tokens']:,} tokens, max {prov['max_tokens']:,}")
    if args.push_to:
        if args.limit is not None:
            raise SystemExit("refusing to push a --limit sample")
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        info = api.upload_folder(
            repo_id=args.push_to,
            repo_type="dataset",
            folder_path=str(out_dir),
            commit_message=f"{args.source}@{source_sha[:8]} rendered with the Olmo 3.5 chat template (plain text, for midtraining)",
        )
        logger.info(f"pushed to {info.commit_url}")


if __name__ == "__main__":
    main()
