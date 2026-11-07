"""Inventory pinned LC releases and prepare explicit context-budget variants.

Reads local snapshots only. Every source row receives a disposition; original
snapshots preserve longer/unsupported rows. No prompt or gold answer is truncated.
Run decon on the emitted questions before publishing the prepared catalog.
"""

import argparse
import json
import multiprocessing
from collections import Counter
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path

from pyarrow import parquet
from transformers import AutoTokenizer

from open_instruct.miles.datasets import long_context
from open_instruct.miles.datasets import source_preparation as sources
from open_instruct.miles.rewards import long_context as reward

RELEASES = {
    "transduction": (
        "nvidia/Nemotron-SFT-Long-Context-Transduction-v1",
        "90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0",
        "lc_transduction_v1",
    ),
    "docqa": ("Tongyi-Zhiwen/DocQA-RL-1.6K", "dbee14eee17e9cf39badb7526a8af316445ca2d5", "lc_docqa_v1"),
    "golong": ("Kwai-Klear/GoLongRL", "34498d935a91b1f39c1e95f9dba6fcac90f04233", "lc_golong_v1"),
}


def records(root, name):
    files = [root / "data/train.jsonl"] if name == "transduction" else sorted(root.rglob("*.parquet"))
    for path in files:
        if path.name.startswith("test"):
            continue
        if path.suffix == ".parquet":
            offset = 0
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    yield path.relative_to(root).as_posix(), offset, row
                    offset += 1
        else:
            with path.open() as stream:
                for offset, line in enumerate(stream):
                    yield path.relative_to(root).as_posix(), offset, json.loads(line)


def control_cases(t):
    if isinstance(t, str):
        return [
            {"kind": "gold", "response": f"</think>\\boxed{{{t}}}", "expected": 1},
            {"kind": "wrong", "response": "</think>\\boxed{918273645}", "expected": 0},
            {"kind": "near_miss", "response": "</think>\\boxed{-918273645}", "expected": 0},
        ]
    metric, answers = t["metric"], t["answers"]
    value = "\n".join(answers) if metric in {"lines_f1", "ndcg"} else answers[0]
    prefix = {
        "final": "",
        "answer_marker": "[Answer]\n",
        "therefore": "Therefore, the answer is ",
        "choice": "The correct answer is ",
    }[t["extraction"]]
    gold = "</think>" + prefix + value
    if metric in {"lines_f1", "ndcg"}:
        near = "\n".join(["__incorrect_item__", *answers[1:]])
    elif metric == "line_accuracy":
        near = "\n".join(["__incorrect_line__", *value.splitlines()[1:]])
    elif metric == "decimal":
        number = Decimal(value)
        near = str(number + max(abs(number) * Decimal("0.1"), Decimal(1)))
    else:
        near = "Z" if metric == "choice" else value + "7"
    wrong = "</think>" + prefix + "__incorrect_answer__"
    near = "</think>" + prefix + near
    return [
        {"kind": "gold", "response": gold, "expected": 1},
        {"kind": "wrong", "response": wrong, "expected": 0},
        {"kind": "near_miss", "response": near, "expected": reward.score(near, t)},
    ]


_WORKER_TOKENIZER = None


def initialize_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)


def analyze_record(item):
    file, index, raw, adapter = item
    try:
        ms, t, verifier, stratum = long_context.adapt(raw, adapter, sources.messages)
        rendered = _WORKER_TOKENIZER.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
        tokens = len(_WORKER_TOKENIZER.encode(rendered, add_special_tokens=False))
        reference = t if isinstance(t, str) else "\n".join(t["answers"])
        response_tokens = len(_WORKER_TOKENIZER.encode(reference, add_special_tokens=False))
        return file, index, raw, (ms, t, stratum, tokens, response_tokens), None
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return file, index, raw, None, str(error)


def inventory(raw_root, output, tokenizer, max_prompt, max_response, workers=1):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "questions").mkdir()
    plan = {"schema_version": 1, "sources": {}}
    summary = {}
    pool_context = (
        multiprocessing.get_context("spawn").Pool(
            workers, initializer=initialize_worker, initargs=(tokenizer.name_or_path,)
        )
        if workers > 1
        else nullcontext(None)
    )
    with (
        pool_context as pool,
        (output / "controls.jsonl").open("w") as controls,
        (output / "dispositions.jsonl").open("w") as dispositions,
    ):
        for name, (repo, revision, adapter) in RELEASES.items():
            root = raw_root / name
            acquisition = json.loads((root / "ACQUISITION.json").read_text())
            if acquisition["repo"] != repo or acquisition["revision"] != revision:
                raise ValueError("Snapshot identity differs from reviewed release")
            for descriptor in acquisition["files"]:
                if sources.file_hash(root / descriptor["path"]) != descriptor["sha256"]:
                    raise ValueError("Downloaded file hash mismatch")
            counts, strata, covered, seen = Counter(), Counter(), set(), {}
            input_path = output / "inputs" / f"{name}.jsonl"
            with input_path.open("w") as stream, (output / "questions" / f"{name}.jsonl").open("w") as questions:
                inputs = ((file, index, raw, adapter) for file, index, raw in records(root, name))
                analyzed = pool.imap(analyze_record, inputs, chunksize=8) if pool else map(analyze_record, inputs)
                for file, index, raw, result, error in analyzed:
                    identity = f"{name}:{file}:{index}"
                    counts["input_rows"] += 1
                    info = {"id": identity, "source": name, "file": file, "row": index}
                    try:
                        if error is not None:
                            raise ValueError(error)
                        ms, t, stratum, tokens, response_tokens = result
                        info.update(subtask=stratum, prompt_tokens=tokens, reference_tokens=response_tokens)
                        counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], tokens)
                        counts["max_reference_tokens"] = max(counts["max_reference_tokens"], response_tokens)
                        if tokens > max_prompt:
                            raise ValueError("deferred_prompt_budget")
                        if response_tokens > max_response:
                            raise ValueError("deferred_reference_exceeds_response_budget")
                        key = sources.digest(ms)
                        if key in seen:
                            if seen[key][0] != sources.digest(t):
                                raise RuntimeError(
                                    "Conflicting duplicate references; resolve both rows before publishing"
                                )
                            info["duplicate_of"] = seen[key][1]
                            raise ValueError("duplicate_prompt")
                        seen[key] = (sources.digest(t), identity)
                        cases = control_cases(t)
                        if cases[-1]["expected"] >= 1:
                            raise ValueError("near_miss_not_distinguished")
                        wrapper = {
                            "id": identity,
                            "raw_record": raw,
                            "provenance": {
                                "repo": repo,
                                "revision": revision,
                                "split": "train",
                                "file": file,
                                "row": index,
                            },
                            "hygiene_status": "pending_reference_check",
                        }
                        stream.write(json.dumps(wrapper, ensure_ascii=False, allow_nan=False) + "\n")
                        questions.write(
                            json.dumps(
                                {"id": identity, "text": "\n".join(m["content"] for m in ms)}, ensure_ascii=False
                            )
                            + "\n"
                        )
                        if stratum not in covered:
                            controls.write(
                                json.dumps({"source": name, "sample_id": identity, "cases": cases}, ensure_ascii=False)
                                + "\n"
                            )
                            covered.add(stratum)
                        strata[stratum] += 1
                        info["status"] = "eligible_pending_hygiene"
                        counts["eligible_rows"] += 1
                    except (ValueError, KeyError, TypeError, IndexError) as error:
                        info.update(status="deferred", reason=str(error))
                        counts[str(error).split(":")[0]] += 1
                    dispositions.write(json.dumps(info, ensure_ascii=False) + "\n")
            summary[name] = {"counts": dict(counts), "strata": dict(strata), "repo": repo, "revision": revision}
            plan["sources"][name] = {
                "adapter": adapter,
                "inputs": [
                    {
                        "path": str(input_path.relative_to(output)),
                        "sha256": sources.file_hash(input_path),
                        "revision": revision,
                        "split": "train",
                    }
                ],
            }
            print(name, dict(counts), flush=True)
    sources.write(
        output / "inventory.json",
        {
            "max_prompt_length": max_prompt,
            "max_reference_tokens": max_response,
            "sources": summary,
            "tokenizer_sha256": sources.file_hash(Path(tokenizer.name_or_path) / "tokenizer.json"),
        },
    )
    sources.write(output / "plan.pending-hygiene.json", plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-reference-tokens", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if min(args.max_prompt_length, args.max_reference_tokens, args.workers) < 1:
        parser.error("Token budgets must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    inventory(args.raw_root, args.output, tokenizer, args.max_prompt_length, args.max_reference_tokens, args.workers)


if __name__ == "__main__":
    main()
