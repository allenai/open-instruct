"""Score HF policy checkpoints on the official GSM8K test split with SGLang engines.

For every ``--checkpoint NAME=HF_DIR`` this serves the checkpoint on ``--gpus`` TP1
engines with the training run's serving settings, then scores all 1,319 test
questions greedily and with ``--sampled-n`` temperature-1 samples each, using the
same ``GSM8KVerifier`` the training run used. Prompts are rendered with the
checkpoint's own chat template exactly as run preparation renders them; when
``--reference-eval`` points at a prepared ``eval.jsonl`` the rendering is checked
against every one of its rows before any generation.

Outputs under ``--output``: ``<name>.jsonl`` with every response, and
``summary.json`` with per-checkpoint accuracy, pass@1/pass@k, truncation and
lengths, plus a paired first-versus-last greedy comparison with an exact McNemar
test on the discordant questions.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
from datasets import load_dataset
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import GSM8KVerifier

logger = logger_utils.setup_logger(__name__)

DATASET = ("openai/gsm8k", "main", "test")
SERVER_FLAGS = [
    "--tp-size",
    "1",
    "--trust-remote-code",
    "--context-length",
    "6144",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "64",
    "--max-total-tokens",
    "524288",
    "--chunked-prefill-size",
    "16384",
    "--max-mamba-cache-size",
    "128",
    "--disable-radix-cache",
    "--attention-backend",
    "triton",
    "--sampling-backend",
    "pytorch",
    "--cuda-graph-max-bs",
    "64",
    "--skip-server-warmup",
]


def parse_label(answer: str) -> str:
    """GSM8K's reference answer ends with ``#### <number>``; labels are comma-free."""
    return answer.split("####")[-1].strip().replace(",", "")


def load_test_set(limit=None):
    rows = load_dataset(DATASET[0], DATASET[1], split=DATASET[2])
    items = [
        {"id": f"gsm8k:test:{index}", "question": row["question"], "label": parse_label(row["answer"])}
        for index, row in enumerate(rows)
    ]
    fingerprint = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    return items[:limit] if limit else items, fingerprint


def render(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def check_rendering(tokenizer, reference: Path) -> int:
    """Every prepared evaluation row must re-render to exactly its recorded prompt."""
    checked = 0
    with reference.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rendered = render(tokenizer, row["metadata"]["query"])
            if rendered != row["input"]:
                raise ValueError(
                    f"Prompt rendering differs from the prepared run data for {row['metadata']['prepared_sample_id']}"
                )
            checked += 1
    return checked


class Engines:
    def __init__(self, model: str, gpus: int, seed: int, log_dir: Path):
        self.model, self.gpus, self.seed, self.log_dir = model, gpus, seed, log_dir
        self.processes: list[subprocess.Popen] = []
        self.urls: list[str] = []

    def start(self):
        env = dict(os.environ, SGLANG_EXTERNAL_MODEL_PACKAGE="olmo_sglang.models")
        for index in range(self.gpus):
            port = 31000 + index
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.model,
                "--port",
                str(port),
                "--base-gpu-id",
                str(index),
                "--random-seed",
                str(self.seed + index),
                *SERVER_FLAGS,
            ]
            log = (self.log_dir / f"engine-{index}.log").open("w")
            self.processes.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            )
            self.urls.append(f"http://127.0.0.1:{port}")
        deadline = time.time() + 20 * 60
        pending = set(self.urls)
        while pending:
            if time.time() > deadline:
                raise RuntimeError(f"Engines did not become healthy: {sorted(pending)}")
            for process in self.processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"An engine exited with {process.returncode} during startup; see {self.log_dir}"
                    )
            for url in list(pending):
                try:
                    if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                        pending.discard(url)
                except httpx.HTTPError:
                    pass
            time.sleep(5)

    def stop(self):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.time() + 120
        for process in self.processes:
            while process.poll() is None and time.time() < deadline:
                time.sleep(1)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        self.processes, self.urls = [], []


async def generate_all(urls, prompts, sampling, concurrency=64):
    """Round-robin prompts across engines; returns one list of completions per prompt."""
    clients = [httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(None)) for url in urls]
    semaphores = [asyncio.Semaphore(concurrency) for _ in urls]
    results = [None] * len(prompts)

    async def one(index, prompt):
        engine = index % len(urls)
        async with semaphores[engine]:
            response = await clients[engine].post("/generate", json={"text": prompt, "sampling_params": sampling})
            response.raise_for_status()
            payload = response.json()
        outputs = payload if isinstance(payload, list) else [payload]
        results[index] = [
            {
                "text": out["text"],
                "finish": out["meta_info"]["finish_reason"]["type"],
                "tokens": out["meta_info"]["completion_tokens"],
            }
            for out in outputs
        ]

    try:
        await asyncio.gather(*(one(index, prompt) for index, prompt in enumerate(prompts)))
    finally:
        for client in clients:
            await client.aclose()
    return results


def score(verifier, text, label):
    return verifier(tokenized_prediction=[], prediction=text, label=label).score


def evaluate_checkpoint(name, model, items, options, output: Path):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rendered_checks = check_rendering(tokenizer, Path(options.reference_eval)) if options.reference_eval else 0
    prompts = [render(tokenizer, item["question"]) for item in items]
    verifier = GSM8KVerifier()
    engines = Engines(model, options.gpus, options.seed, output / f"{name}-engine-logs")
    engines.log_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    started = time.perf_counter()
    engines.start()
    timings["engine_startup_seconds"] = time.perf_counter() - started
    try:
        started = time.perf_counter()
        greedy = asyncio.run(
            generate_all(engines.urls, prompts, {"temperature": 0.0, "max_new_tokens": options.max_new_tokens})
        )
        timings["greedy_seconds"] = time.perf_counter() - started
        sampled = None
        if options.sampled_n > 0:
            started = time.perf_counter()
            sampled = asyncio.run(
                generate_all(
                    engines.urls,
                    prompts,
                    {"temperature": 1.0, "n": options.sampled_n, "max_new_tokens": options.max_new_tokens},
                )
            )
            timings["sampled_seconds"] = time.perf_counter() - started
    finally:
        engines.stop()

    records = []
    with (output / f"{name}.jsonl").open("w") as handle:
        for index, item in enumerate(items):
            g = greedy[index][0]
            record = {
                "id": item["id"],
                "label": item["label"],
                "greedy": {**g, "correct": score(verifier, g["text"], item["label"])},
            }
            if sampled is not None:
                record["sampled"] = [
                    {**s, "correct": score(verifier, s["text"], item["label"])} for s in sampled[index]
                ]
            records.append(record)
            handle.write(json.dumps(record) + "\n")

    n = len(records)
    summary = {
        "model": model,
        "questions": n,
        "rendering_checked_against_prepared_rows": rendered_checks,
        "timings": timings,
        "greedy": {
            "correct": int(sum(r["greedy"]["correct"] for r in records)),
            "accuracy": sum(r["greedy"]["correct"] for r in records) / n,
            "truncated": sum(r["greedy"]["finish"] == "length" for r in records) / n,
            "mean_tokens": statistics.mean(r["greedy"]["tokens"] for r in records),
        },
    }
    if sampled is not None:
        k = options.sampled_n
        per_q = [sum(s["correct"] for s in r["sampled"]) / k for r in records]
        summary["sampled"] = {
            "n": k,
            "pass_at_1": statistics.mean(per_q),
            f"pass_at_{k}": sum(any(s["correct"] for s in r["sampled"]) for r in records) / n,
            "all_correct_fraction": sum(all(s["correct"] for s in r["sampled"]) for r in records) / n,
            "all_wrong_fraction": sum(not any(s["correct"] for s in r["sampled"]) for r in records) / n,
            "truncated": statistics.mean(s["finish"] == "length" for r in records for s in r["sampled"]),
            "mean_tokens": statistics.mean(s["tokens"] for r in records for s in r["sampled"]),
        }
    return summary, records


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs (b: only first, c: only second)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(first, second):
    a = {r["id"]: r["greedy"]["correct"] for r in first}
    b = {r["id"]: r["greedy"]["correct"] for r in second}
    both = sum(1 for i in a if a[i] and b[i])
    only_first = sum(1 for i in a if a[i] and not b[i])
    only_second = sum(1 for i in a if not a[i] and b[i])
    neither = sum(1 for i in a if not a[i] and not b[i])
    return {
        "both_correct": both,
        "only_first": only_first,
        "only_second": only_second,
        "neither": neither,
        "net_change": only_second - only_first,
        "mcnemar_exact_p": mcnemar_exact(only_first, only_second),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=HF_DIR, in order")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampled-n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke runs)")
    parser.add_argument("--reference-eval", default=None, help="Prepared eval.jsonl to check prompt rendering")
    options = parser.parse_args()

    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    items, fingerprint = load_test_set(options.limit)
    logger.info("GSM8K test: %d questions, fingerprint %s", len(items), fingerprint[:16])
    summary = {
        "dataset": {"source": DATASET, "questions": len(items), "fingerprint_sha256": fingerprint},
        "settings": {
            "gpus": options.gpus,
            "sampled_n": options.sampled_n,
            "max_new_tokens": options.max_new_tokens,
            "server_flags": SERVER_FLAGS,
        },
        "checkpoints": {},
    }
    all_records = {}
    for spec in options.checkpoint:
        name, model = spec.split("=", 1)
        logger.info("Evaluating %s: %s", name, model)
        summary["checkpoints"][name], all_records[name] = evaluate_checkpoint(name, model, items, options, output)
        logger.info("%s: %s", name, json.dumps(summary["checkpoints"][name]["greedy"]))
        with (output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
    names = list(all_records)
    if len(names) >= 2:
        summary["paired_greedy_first_vs_last"] = {"first": names[0], "last": names[-1]} | paired(
            all_records[names[0]], all_records[names[-1]]
        )
        with (output / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
