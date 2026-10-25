"""Prepare benchmark-only long prompts from captured natural continuations."""

import argparse
import hashlib
import json
from pathlib import Path

from open_instruct.miles.datasets import run_data
from open_instruct.miles.rollout import benchmark_metrics


class BenchmarkOnlyVerifier:
    def __init__(self, *args, **kwargs):
        raise ValueError("Continuation workload requires benchmark.reward_mode=none; it has no reward targets")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def continuation_prefix(record, tokenizer, target, slack):
    """Find an exact text/token boundary without fabricating or padding tokens."""
    for length in range(target, max(record["prompt_tokens"], target - slack - 1), -1):
        ids = record["tokens"][:length]
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) == ids:
            return text, ids
    return None


def prepare(model, output, references, *, prompt_tokens=8192, boundary_slack=32, remaining_tokens=2048, seed=17):
    if prompt_tokens < 1 or boundary_slack < 0 or remaining_tokens < 1:
        raise ValueError("Prompt/remaining lengths must be positive and boundary slack nonnegative")
    output, model = Path(output), Path(model).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory for an immutable continuation workload")
    tokenizer = run_data._tokenizer(model)
    sources, selected, excluded = [], {}, []
    model_contract = None
    for reference in sorted({Path(path).resolve() for path in references}):
        spec = json.loads((reference / "run-spec.json").read_text())
        if Path(spec["model"]["source"]).resolve() != model:
            raise ValueError("Continuation source must use the requested checkpoint/tokenizer")
        if model_contract is not None and spec["model"] != model_contract:
            raise ValueError("Continuation sources have different model contracts")
        model_contract = spec["model"]
        raw, records = benchmark_metrics.reference_records(
            reference, model_contract, spec["inference"]["max_context_length"]
        )
        reference_sha = hashlib.sha256(raw).hexdigest()
        sources.append({"path": str(reference), "captures_sha256": reference_sha, "captures": len(records)})
        for record in records:
            origin = {
                "reference_sha256": reference_sha,
                "sample_index": record["index"],
                "group_index": record["group_index"],
                "original_prompt_tokens": record["prompt_tokens"],
                "original_response_tokens": record["response_tokens"],
                "trajectory_sha256": digest(record["tokens"]),
            }
            if record["prompt_tokens"] >= prompt_tokens or len(record["tokens"]) < prompt_tokens + remaining_tokens:
                excluded.append({**origin, "reason": "insufficient_continuation"})
                continue
            prefix = continuation_prefix(record, tokenizer, prompt_tokens, boundary_slack)
            if prefix is None:
                excluded.append({**origin, "reason": "no_exact_text_roundtrip_boundary"})
                continue
            text, ids = prefix
            identity = digest(ids)
            if identity in selected:
                selected[identity]["metadata"]["continuation_origins"].append(origin)
                continue
            selected[identity] = {
                "input": text,
                "metadata": {
                    "prepared_sample_id": f"continuation:{identity}",
                    "benchmark_only": True,
                    "workload": "captured_natural_continuation",
                    "prompt_token_ids": ids,
                    "original_prompt_sha256": digest(record["tokens"][: record["prompt_tokens"]]),
                    "continuation_origins": [origin],
                    "verifiers": [{"name": "benchmark_only", "target": None}],
                },
            }
    if not selected:
        raise ValueError("No captured continuations satisfy the requested lengths and exact tokenizer roundtrip")
    rows = sorted(selected.values(), key=lambda row: digest([seed, row["metadata"]["prepared_sample_id"]]))
    source = output / "source"
    source.mkdir(parents=True)
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "verifiers.json").write_text(
        json.dumps(
            {"benchmark_only": {"factory": "scripts.miles.prepare_inference_continuations.BenchmarkOnlyVerifier"}}
        )
    )
    selection = {
        "schema_version": 1,
        "workload": "captured_natural_continuation",
        "benchmark_only": True,
        "scope": "Long prefill mechanics; continuations sharing an original prompt are not independent prompts",
        "seed": seed,
        "target_prompt_tokens": prompt_tokens,
        "boundary_slack": boundary_slack,
        "minimum_original_remaining_tokens": remaining_tokens,
        "model": model_contract,
        "sources": sources,
        "selected_rows": len(rows),
        "original_prompt_clusters": len({row["metadata"]["original_prompt_sha256"] for row in rows}),
        "actual_prompt_lengths": [len(row["metadata"]["prompt_token_ids"]) for row in rows],
        "excluded": excluded,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    result = run_data.prepare_data(
        {"prompt_data": str(source / "train.jsonl"), "reward_config": str(source / "verifiers.json")},
        model,
        output / "prepared",
        max_prompt_length=prompt_tokens,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--boundary-slack", type=int, default=32)
    parser.add_argument("--remaining-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(
        args.model,
        args.output,
        args.reference,
        prompt_tokens=args.prompt_tokens,
        boundary_slack=args.boundary_slack,
        remaining_tokens=args.remaining_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
