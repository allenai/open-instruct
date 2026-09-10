"""Prepare nvidia/Nemotron-SFT-Agentic-v2's `tool_calling` split for SFT through open-instruct.

The split is one 15 GB JSONL file whose `tools` column holds every function's JSON schema as
nested objects. Schema fields such as `default` take different types across rows, so Arrow's
JSON reader (which `datasets` uses) fails on it:

    ArrowInvalid: JSON parse error: Column(/tools/[]/function/parameters/properties/limit/default)
    changed from string to number in row 2

This script reads the file line by line instead, and writes parquet shards in the layout the SFT
tokenizer already handles: `messages` as structs, `tools` as a JSON string (which
`_normalize_tools_for_chat_template` parses), `tool_calls[].function.arguments` left as the JSON
string it already is (parsed by `_normalize_tool_call_arguments`), and the provenance objects
serialized as JSON strings. Along the way it applies two filters chosen for the Olmo 3.5 runs:

* rows whose full render under the Olmo 3.5 chat template exceeds --max-tokens are dropped
  (about 1% of the split, almost all of it giant tool lists: one sampled row was 102,823 tokens,
  99,857 of them schemas);
* system messages with empty content are removed, so the template's own default applies instead
  of an empty system line before the tools block.

Row order is preserved. Usage (0 GPUs, many CPUs):

    python scripts/data/convert_nemotron_agentic_tool_calling.py \\
        --out-dir /weka/oe-adapt-default/<user>/nemotron-agentic-v2-tool-calling \\
        --push-to allenai/nemotron-sft-agentic-v2-tool-calling-oi --private
"""

import argparse
import json
import multiprocessing
import os
import pathlib
import time
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

SOURCE_REPO = "nvidia/Nemotron-SFT-Agentic-v2"
SOURCE_FILE = "data/tool_calling.jsonl"
DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer-olmo35"

TOOL_CALL = pa.struct(
    [
        ("id", pa.string()),
        ("type", pa.string()),
        ("function", pa.struct([("name", pa.string()), ("arguments", pa.string())])),
    ]
)
MESSAGE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.string()),
        ("reasoning_content", pa.string()),
        ("tool_calls", pa.list_(TOOL_CALL)),
        ("tool_call_id", pa.string()),
    ]
)
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("messages", pa.list_(MESSAGE)),
        ("tools", pa.string()),
        ("metadata", pa.string()),
        ("n_tokens", pa.int32()),
    ]
)

_tokenizer = None


def _get_tokenizer(name: str, revision: str):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    return _tokenizer


def _parses_to_null(arguments: str) -> bool:
    try:
        return json.loads(arguments) is None
    except json.JSONDecodeError:
        return False


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = None
    if message.get("tool_calls"):
        tool_calls = []
        for tc in message["tool_calls"]:
            function = tc.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
            elif _parses_to_null(arguments):
                # A handful of source rows carry the JSON string "null" for a no-argument call. Chat
                # templates iterate arguments as a mapping, and the SFT tokenizer only parses strings
                # that decode to an object, so "null" would reach the template as a string and crash
                # it (jinja: "Can only get item pairs from a mapping"). Store the empty object instead.
                arguments = "{}"
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "type": tc.get("type") or "function",
                    "function": {"name": function.get("name"), "arguments": arguments},
                }
            )
    return {
        "role": message["role"],
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content") or None,
        "tool_calls": tool_calls,
        "tool_call_id": message.get("tool_call_id"),
    }


def _render_for_count(messages: list[dict[str, Any]], tools: list | None, tokenizer) -> int:
    # Builds fresh dicts: the rows are written as-is afterwards, with arguments still a JSON string.
    chat = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m.get("tool_calls"):
            m["tool_calls"] = [
                dict(tc, function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"])))
                for tc in m["tool_calls"]
            ]
        chat.append(m)
    text = tokenizer.apply_chat_template(chat, tools=tools, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def process_chunk(args: tuple[list[str], str, str, int]) -> tuple[list[dict[str, Any]], Counter]:
    lines, tokenizer_name, tokenizer_revision, max_tokens = args
    tokenizer = _get_tokenizer(tokenizer_name, tokenizer_revision)
    out, stats = [], Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        stats["rows_in"] += 1
        messages = [_clean_message(m) for m in row["messages"]]
        if messages and messages[0]["role"] == "system" and not (messages[0]["content"] or "").strip():
            messages = messages[1:]
            stats["empty_system_removed"] += 1
        tools = row.get("tools") or None
        try:
            n_tokens = _render_for_count(messages, tools, tokenizer)
        except Exception as exc:  # a row the template cannot render is not usable downstream either
            stats["dropped_unrenderable"] += 1
            logger.warning(f"dropping unrenderable row: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if n_tokens > max_tokens:
            stats["dropped_over_length"] += 1
            continue
        metadata = row.get("metadata") or {}
        out.append(
            {
                "id": str(metadata.get("uuid") or metadata.get("alt_id") or ""),
                "source": str(metadata.get("source") or ""),
                "messages": messages,
                "tools": json.dumps(tools, ensure_ascii=False) if tools else None,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "n_tokens": n_tokens,
            }
        )
        stats["rows_out"] += 1
        stats["tokens_out"] += n_tokens
    return out, stats


def iter_chunks(path: str, chunk_lines: int):
    chunk: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_lines:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _limit_chunks(chunks, limit: int):
    """Yield at most `limit` lines in total (smoke tests)."""
    budget = limit
    for chunk in chunks:
        if budget <= 0:
            return
        yield chunk[:budget]
        budget -= len(chunk)


def write_readme(
    out_dir: pathlib.Path, stats: Counter, args: argparse.Namespace, source_sha: str, tokenizer_sha: str
) -> None:
    data_size = sum(p.stat().st_size for p in (out_dir / "data").glob("*.parquet"))
    readme = f"""---
license: cc-by-4.0
dataset_info:
  features:
  - name: id
    dtype: string
  - name: source
    dtype: string
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
    - name: reasoning_content
      dtype: string
    - name: tool_calls
      list:
      - name: id
        dtype: string
      - name: type
        dtype: string
      - name: function
        struct:
        - name: name
          dtype: string
        - name: arguments
          dtype: string
    - name: tool_call_id
      dtype: string
  - name: tools
    dtype: string
  - name: metadata
    dtype: string
  - name: n_tokens
    dtype: int32
  splits:
  - name: train
    num_examples: {stats["rows_out"]}
  download_size: {data_size}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Nemotron-SFT-Agentic-v2, `tool_calling` split, prepared for open-instruct SFT

Derived from [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}) (revision `{source_sha}`),
`{SOURCE_FILE}`. Same conversations and text; the layout and two filters differ:

- `tools` is a JSON string. In the source it is a list of nested schema objects whose fields take
  different types across rows, which Arrow's JSON reader rejects, so the source cannot be loaded
  with `datasets` directly.
- `metadata` (uuid, source, generator model, judge scores) is a JSON string; `id` and `source`
  are lifted out of it. The source's `processing_info`, `filter_reason`, `match_contexts` and
  `matched_categories` columns are dropped (`filter_reason` was null on every sampled row).
- Rows whose full render under the Olmo 3.5 chat template
  ([{args.tokenizer}](https://huggingface.co/{args.tokenizer}) @ `{tokenizer_sha}`) exceeds
  {args.max_tokens:,} tokens are dropped: {stats["dropped_over_length"]:,} rows, almost all of them
  giant tool lists. `n_tokens` is that render length for the rows kept.
- System messages with empty content are removed ({stats["empty_system_removed"]:,} rows) so a chat
  template's default system prompt applies.
- `messages` fields are `role`, `content`, `reasoning_content` (assistant only), `tool_calls`
  (assistant only; `arguments` is a JSON string, as in the source, except that the JSON string
  `null` becomes `{{}}` so templates that iterate arguments as a mapping accept it) and
  `tool_call_id` (tool only).

| | |
|---|---|
| rows in | {stats["rows_in"]:,} |
| rows out | {stats["rows_out"]:,} |
| dropped, over {args.max_tokens:,} tokens | {stats["dropped_over_length"]:,} |
| dropped, unrenderable | {stats["dropped_unrenderable"]:,} |
| tokens out | {stats["tokens_out"]:,} |

License follows the source: CC-BY-4.0, with Apache 2.0 and MIT components.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--chunk-lines", type=int, default=500)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", os.cpu_count() or 8))) - 2),
    )
    parser.add_argument(
        "--input-path", default=None, help="Local JSONL to read instead of downloading the split (tests)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    parser.add_argument("--push-to", default=None, help="HF dataset repo to upload the result to.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    source_sha = api.dataset_info(SOURCE_REPO).sha
    tokenizer_sha = api.model_info(args.tokenizer).sha
    logger.info(f"source {SOURCE_REPO}@{source_sha}, tokenizer {args.tokenizer}@{tokenizer_sha}")
    path = args.input_path or hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=source_sha)
    logger.info(f"input {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    pending: list[dict[str, Any]] = []
    shard = 0
    t0 = time.time()

    def flush(force: bool = False) -> None:
        nonlocal pending, shard
        while len(pending) >= args.rows_per_shard or (force and pending):
            rows, pending = pending[: args.rows_per_shard], pending[args.rows_per_shard :]
            dst = out_dir / "data" / f"train-{shard:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), dst, compression="zstd")
            logger.info(f"wrote {dst.name}: {len(rows)} rows")
            shard += 1

    chunks = iter_chunks(path, args.chunk_lines)
    if args.limit is not None:
        chunks = _limit_chunks(chunks, args.limit)
    tasks = ((c, args.tokenizer, tokenizer_sha, args.max_tokens) for c in chunks)
    with multiprocessing.get_context("spawn").Pool(args.num_proc) as pool:
        for i, (rows, chunk_stats) in enumerate(pool.imap(process_chunk, tasks, chunksize=1)):
            pending.extend(rows)
            stats.update(chunk_stats)
            flush()
            if i % 100 == 0:
                logger.info(f"{stats['rows_in']:,} rows in, {stats['rows_out']:,} out, {time.time() - t0:.0f}s")
    flush(force=True)
    # rename shards to the conventional train-XXXXX-of-NNNNN form
    shards = sorted((out_dir / "data").glob("train-*.parquet"))
    for i, p in enumerate(shards):
        p.rename(p.with_name(f"train-{i:05d}-of-{len(shards):05d}.parquet"))
    write_readme(out_dir, stats, args, source_sha, tokenizer_sha)
    (out_dir / "stats.json").write_text(json.dumps(dict(stats), indent=1))
    logger.info(f"done in {time.time() - t0:.0f}s: {json.dumps(dict(stats))}")

    if args.push_to:
        api.create_repo(args.push_to, repo_type="dataset", private=args.private, exist_ok=True)
        info = api.upload_folder(
            repo_id=args.push_to,
            repo_type="dataset",
            folder_path=str(out_dir),
            commit_message=f"{SOURCE_REPO} tool_calling split for open-instruct SFT (max {args.max_tokens} tokens, empty system messages removed)",
        )
        logger.info(f"pushed to {info.commit_url}")


if __name__ == "__main__":
    main()
