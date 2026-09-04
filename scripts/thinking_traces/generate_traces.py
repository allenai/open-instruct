"""Generate thinking traces against a local vLLM server and record their lengths.

This is the client half of the thinking-trace-length experiment. It samples a
small, deterministic slice of a post-training SFT mixture (default:
``allenai/Dolci-Think-SFT-7B``), replays each prompt through an OpenAI-compatible
vLLM endpoint ``--num-samples`` times, splits every completion into its
``<think>...</think>`` trace and its final answer, and writes one JSON line per
generated trace with exact token counts.

The sampling is seeded and depends only on (dataset, revision, seed,
num_prompts, max_prompt_tokens), so two jobs serving two different models select
byte-identical prompts. Each record carries ``prompt_sha`` so the analysis step
can verify that rather than assume it.

Records stream to disk as they complete, so a preempted job still yields usable
partial results.

Example:
    PYTHONPATH=. uv run python scripts/thinking_traces/generate_traces.py \\
        --model qwen3-8b --tokenizer Qwen/Qwen3-8B \\
        --api-base http://localhost:8008/v1 \\
        --num-prompts 200 --num-samples 4 --output /results/traces.jsonl
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import logging
import os
import threading
import time

import datasets
import huggingface_hub
import openai
import transformers

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


class SuppressHTTPChatter(logging.Filter):
    """Drop the per-request INFO logs the HTTP client libraries emit.

    Generating a few thousand traces means a few thousand "HTTP Request: POST"
    lines, and streaming a dataset adds one long presigned-URL line per parquet
    range request. Together they bury the progress output in the Beaker log.

    Filtering on the logger-name prefix rather than setting levels on a fixed
    list of loggers matters: the name is not always the one you expect (some
    environments ship a vendored ``httpx2``), and loggers created after startup
    would miss a one-shot setLevel pass. Warnings and errors still get through,
    so a genuinely failing request is still visible.
    """

    NOISY_PREFIXES = ("httpx", "httpcore", "urllib3", "openai", "filelock", "fsspec")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return not record.name.startswith(self.NOISY_PREFIXES)


for _handler in logging.getLogger().handlers:
    _handler.addFilter(SuppressHTTPChatter())

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Trace shapes we can tell apart from the completion text plus finish_reason.
KIND_CLOSED = "closed"  # a complete <think>...</think> block: length is exact
KIND_TRUNCATED = "truncated"  # hit the token cap mid-trace: length is censored
KIND_NO_BLOCK = "no_block"  # model answered without thinking: length is zero


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="served-model-name to request from vLLM")
    parser.add_argument("--tokenizer", required=True, help="HF repo whose tokenizer counts the tokens")
    parser.add_argument("--api-base", default="http://localhost:8008/v1")
    parser.add_argument("--dataset", default="allenai/Dolci-Think-SFT-7B")
    parser.add_argument("--parse-health-after", type=int, default=50, help="log the trace-shape mix after N traces")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=4, help="completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=30720)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--output", required=True, help="destination .jsonl")
    parser.add_argument("--prompts-output", default=None, help="optional .jsonl copy of the resolved prompt sample")
    return parser.parse_args()


def dataset_revision(name: str) -> str | None:
    """Resolve the corpus to an immutable commit sha.

    Without this the join back to the source dataset is only as stable as the
    Hub's main branch: a re-upload would silently repoint every source_id.
    """
    try:
        return huggingface_hub.dataset_info(name).sha
    except Exception as exc:  # noqa: BLE001 - a missing revision must not stop a run
        logger.warning("could not resolve revision for %s: %s", name, exc)
        return None


def select_prompts(args: argparse.Namespace, tokenizer) -> list[dict]:
    """Deterministically pick ``--num-prompts`` single-turn prompts from the mixture.

    Streams the dataset so we never materialize a multi-gigabyte mixture to pick
    a couple hundred rows, and shuffles with a fixed seed so every model in the
    comparison sees the same prompts in the same order.
    """
    stream = datasets.load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    prompts: list[dict] = []
    seen_shas: set[str] = set()
    scanned = 0
    for row in stream:
        scanned += 1
        messages = row.get("messages") or []
        # The mixture stores full conversations; we replay only the opening user
        # turn so every model starts from the same, unconditioned state.
        if not messages or messages[0].get("role") != "user":
            continue
        text = (messages[0].get("content") or "").strip()
        if not text:
            continue
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha in seen_shas:
            continue
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > args.max_prompt_tokens:
            continue
        seen_shas.add(sha)
        prompts.append(
            {
                "prompt_index": len(prompts),
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
                "prompt_sha": sha,
                "prompt": text,
                "prompt_tokens": n_tokens,
                "dataset_source": row.get("dataset_source"),
                "source_id": row.get("id"),
            }
        )
        if len(prompts) >= args.num_prompts:
            break

    logger.info("selected %d prompts after scanning %d rows", len(prompts), scanned)
    if len(prompts) < args.num_prompts:
        logger.warning("dataset exhausted with only %d/%d prompts", len(prompts), args.num_prompts)
    return prompts


def split_trace(text: str, finish_reason: str) -> tuple[str, str, str]:
    """Split a completion into (thinking_text, answer_text, kind).

    Two chat templates behave differently and both must work:
      * Qwen3 lets the model emit its own opening ``<think>``.
      * DeepSeek-R1-Distill prefills ``<think>`` in the assistant prefix, so the
        completion starts *inside* the trace with no opening tag of its own.

    So the closing tag, not the opening one, is what marks a complete trace.
    """
    close_at = text.find(THINK_CLOSE)
    if close_at != -1:
        head = text[:close_at]
        open_at = head.find(THINK_OPEN)
        if open_at != -1:
            head = head[open_at + len(THINK_OPEN) :]
        return head, text[close_at + len(THINK_CLOSE) :], KIND_CLOSED

    # No closing tag. If we ran out of budget the trace is real but censored;
    # otherwise the model simply answered without opening a thinking block.
    if finish_reason == "length":
        open_at = text.find(THINK_OPEN)
        body = text[open_at + len(THINK_OPEN) :] if open_at != -1 else text
        return body, "", KIND_TRUNCATED
    open_at = text.find(THINK_OPEN)
    if open_at != -1:
        return text[open_at + len(THINK_OPEN) :], "", KIND_TRUNCATED
    return "", text, KIND_NO_BLOCK


def generate_one(client, args, tokenizer, prompt: dict, sample_index: int) -> dict:
    """Request a single completion and measure its trace, retrying transient errors."""
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            started = time.monotonic()
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt["prompt"]}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                # Distinct per (prompt, sample) so the run is reproducible while
                # the samples within a prompt stay independent draws.
                seed=args.seed * 1_000_003 + prompt["prompt_index"] * 97 + sample_index,
                n=1,
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry anything the server throws
            last_error = exc
            if attempt == args.max_retries:
                logger.error(
                    "prompt %d sample %d failed after %d attempts: %s",
                    prompt["prompt_index"],
                    sample_index,
                    attempt + 1,
                    exc,
                )
                return {
                    "prompt_index": prompt["prompt_index"],
                    "prompt_sha": prompt["prompt_sha"],
                    "sample_index": sample_index,
                    "model": args.model,
                    "error": str(exc),
                }
            time.sleep(min(2**attempt, 30))
    else:  # pragma: no cover - the loop always breaks or returns
        raise RuntimeError(f"unreachable: {last_error}")

    choice = response.choices[0]
    text = choice.message.content or ""
    # Some builds route the trace to reasoning_content when a reasoning parser
    # is enabled; stitch it back so parsing sees one canonical string.
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning:
        text = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{text}"

    thinking_text, answer_text, kind = split_trace(text, choice.finish_reason)
    thinking_tokens = len(tokenizer(thinking_text, add_special_tokens=False)["input_ids"])
    answer_tokens = len(tokenizer(answer_text, add_special_tokens=False)["input_ids"])
    usage = response.usage

    return {
        "prompt_index": prompt["prompt_index"],
        "prompt_sha": prompt["prompt_sha"],
        "sample_index": sample_index,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset": prompt["dataset"],
        "dataset_revision": prompt["dataset_revision"],
        "source_id": prompt["source_id"],
        "dataset_source": prompt["dataset_source"],
        "prompt_tokens": prompt["prompt_tokens"],
        "kind": kind,
        "truncated": kind == KIND_TRUNCATED,
        "finish_reason": choice.finish_reason,
        "thinking_tokens": thinking_tokens,
        "answer_tokens": answer_tokens,
        "thinking_chars": len(thinking_text),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "latency_s": round(time.monotonic() - started, 3),
        # Full text, not an excerpt: these traces are the expensive artifact of
        # the run and get published as a dataset. Storing them whole also means
        # a parsing change can be re-applied offline instead of re-generating.
        "thinking_text": thinking_text,
        "answer_text": answer_text,
    }


def main() -> None:
    args = parse_args()
    logger.info("loading tokenizer %s", args.tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    args.dataset_revision = dataset_revision(args.dataset)
    logger.info("corpus %s pinned at revision %s", args.dataset, args.dataset_revision)
    prompts = select_prompts(args, tokenizer)
    if args.prompts_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.prompts_output)), exist_ok=True)
        with open(args.prompts_output, "w") as handle:
            for prompt in prompts:
                handle.write(json.dumps(prompt) + "\n")

    client = openai.OpenAI(
        base_url=args.api_base,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        timeout=args.request_timeout,
        max_retries=0,  # retries are handled in generate_one so they get logged
    )

    work = [(prompt, sample) for prompt in prompts for sample in range(args.num_samples)]
    logger.info("generating %d traces (%d prompts x %d samples)", len(work), len(prompts), args.num_samples)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_lock = threading.Lock()
    shapes: collections.Counter = collections.Counter()
    done = 0
    started_at = time.monotonic()

    with open(args.output, "w") as handle, concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, args, tokenizer, p, s) for p, s in work]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            with write_lock:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                done += 1
                shapes[record.get("kind") or "error"] += 1
                # Early warning: if a chat template behaves differently than the
                # parser expects, the shape mix is wrong from the first handful
                # of traces. Surfacing it here turns a 14-hour discovery into a
                # 5-minute one -- and because full text is stored, a mis-parse is
                # recoverable offline rather than needing the run repeated.
                if done == args.parse_health_after:
                    logger.info("parse health after %d traces: %s", done, dict(shapes))
                    if shapes.get(KIND_CLOSED, 0) == 0:
                        logger.error(
                            "NO closed <think>...</think> traces yet -- the parser and this "
                            "model's chat template likely disagree. Full text is being stored, "
                            "so this is fixable offline, but check before trusting the lengths."
                        )
                if done % 25 == 0 or done == len(work):
                    elapsed = time.monotonic() - started_at
                    logger.info(
                        "%d/%d traces done (%.1f min elapsed, %.1f traces/min)",
                        done,
                        len(work),
                        elapsed / 60,
                        done / max(elapsed / 60, 1e-9),
                    )

    logger.info("final trace-shape mix: %s", dict(shapes))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
