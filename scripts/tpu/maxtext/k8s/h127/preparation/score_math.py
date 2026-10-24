"""Score complete matched H127 MATH generations with the historical MATH harness."""

import argparse
import asyncio
import gzip
import hashlib
import json
import math
from pathlib import Path

from olmo_eval.common.execution import ProcessPoolManager, ProcessScoringPoolConfig, ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.runners.io.builders import build_predictions

REQUESTS_SHA = "c126484d6be7a8a0271a9d4bcdc52ff1629ddd3d6a81b5e05ea028f6f61d4595"
FIXTURES_SHA = "438624274108a5856df7314620489ebe80308f28de6372b486478dce5760f11a"


def read(path):
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == ".gz" else json.loads(path.read_text())


async def score_one(item, text, context):
    assert item["task"] == "MATH"
    row = item["row"]
    task = get_task(row["task_name"] + ":olmo3adapt")
    instance = Instance(
        question=row["doc"]["query"],
        gold_answer=row["label"],
        metadata={k: v for k, v in row["doc"].items() if k != "query"},
    )
    response = Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=""),
        outputs=[LMOutput(text=text)],
    )
    assert task.config.strip_thinking
    task.strip_thinking_traces([response])
    scored = await task.score_responses([response], context=context)
    prediction = build_predictions(scored, task.config.metrics)[0]
    values = prediction["instance_metrics"]["accuracy"]
    assert len(values) == 1
    value = next(iter(values.values()))
    assert math.isfinite(value) and value in (0, 1)
    assert not any(o.get("scoring_errors") for o in prediction["model_output"]), (
        "Scoring failure is not a model failure"
    )
    return {"score": value, "extracted_answer": response.outputs[0].extracted_answer}


def validate_generations(paths, requests, panel):
    records = {}
    fingerprints = set()
    metadata = []
    for folder in paths:
        manifest = read(folder / "generation-complete.json")
        assert manifest["pilot_items"] is None, "A pilot is not a complete quality evaluation"
        fingerprints.add(manifest["fingerprint"])
        metadata.append(manifest)
        for row in manifest["rows"]:
            path = folder / row["file"]
            assert path.parent == folder
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            record = read(path)
            qid = record["qualified_id"]
            assert qid == row["qualified_id"] and qid in requests and qid not in records
            assert record["fingerprint"] == manifest["fingerprint"]
            assert record["sampling"] == manifest["sampling"]
            assert record["arm"] == manifest["arm"] and record["step"] == manifest["step"] == 2048
            assert record["input_ids"] == panel[qid]["input_ids"], "Generated prompt differs from frozen primary"
            assert record["input_ids"][-1] == 100256
            assert record["finish_reason"]["type"] in ("stop", "length")
            assert len(record["body_ids"]) <= 32768
            records[qid] = record
    assert len(fingerprints) == 1 and set(records) == set(requests), "Incomplete or mixed generation run"
    sampling = metadata[0]["sampling"]
    assert sampling == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "max_new_tokens": 32768,
        "stop_token_ids": [100257, 100265],
        "skip_special_tokens": True,
        "no_stop_trim": False,
    }
    return records, metadata


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--generations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    assert hashlib.sha256(args.requests.read_bytes()).hexdigest() == REQUESTS_SHA
    assert hashlib.sha256(args.fixtures.read_bytes()).hexdigest() == FIXTURES_SHA
    requests = read(args.requests)
    assert len(requests) == 1000
    args.output.mkdir(parents=True, exist_ok=False)
    pool = ProcessPoolManager({"cpu": ProcessScoringPoolConfig(workers=4)})
    context = ScoringContext(process_pool_manager=pool)
    try:
        fixtures = read(args.fixtures)
        for row in fixtures:
            result = await score_one(row["item"], row["text"], context)
            assert result["score"] == row["expected_score"]
        (args.output / "scorer-validation.json").write_text(
            json.dumps({"passed": True, "historical_cases": len(fixtures)})
        )
        if args.validate_only:
            print("MATH-SCORER-VALIDATED", len(fixtures), flush=True)
            return
        assert args.panel is not None and args.generations
        assert (
            hashlib.sha256(args.panel.read_bytes()).hexdigest()
            == "96d18a968061fa88aae38ad1dd69ac297345db5ada3994522b6ca5de034dfecc"
        )
        with gzip.open(args.panel, "rt") as stream:
            panel = {r["qualified_id"]: r for r in (json.loads(line) for line in stream)}
        assert set(panel) == set(requests)
        records, manifests = validate_generations(args.generations, requests, panel)
        output = []
        for qid, record in sorted(records.items()):
            text = record["text"]
            historical = await score_one(requests[qid], text, context)
            close_count = text.count("</think>")
            closed = close_count == 1
            final = text.split("</think>", 1)[1].strip() if closed else ""
            final_score = await score_one(requests[qid], final, context) if final else {"score": 0}
            body = record["body_ids"]
            close_position = next(
                (i for i, t in enumerate(body) if t == 100270 or (t == 524 and body[i + 1 : i + 2] == [27963])), None
            )
            assert not closed or close_position is not None
            output.append(
                {
                    "qualified_id": qid,
                    "arm": record["arm"],
                    "step": record["step"],
                    "accuracy": historical["score"],
                    "explicit_final_accuracy": final_score["score"],
                    "cap_hit": record["finish_reason"]["type"] == "length",
                    "closed": closed,
                    "close_count": close_count,
                    "atomic_close_count": body.count(100270),
                    "extraction_empty": not bool(historical["extracted_answer"]),
                    "body_tokens": len(body),
                    "thinking_tokens": close_position if close_position is not None else len(body),
                }
            )
            with (args.output / "scores.jsonl").open("a") as stream:
                stream.write(json.dumps(output[-1], allow_nan=False) + "\n")
            if len(output) % 100 == 0:
                print("MATH-SCORED", len(output), "of", len(records), flush=True)
        summary = {
            "n": len(output),
            "arm": output[0]["arm"],
            "step": 2048,
            "means": {
                k: sum(float(r[k]) for r in output) / len(output)
                for k in (
                    "accuracy",
                    "explicit_final_accuracy",
                    "cap_hit",
                    "closed",
                    "extraction_empty",
                    "body_tokens",
                    "thinking_tokens",
                )
            },
            "generation_manifests": manifests,
            "scorer": "allenai/olmo-eval@6f4cbee81ffbd79b210b92d6adb7e37744613020; minerva_math_*:olmo3adapt",
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    finally:
        pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
