"""Score an exported checkpoint the way the EOPD paper's Table 2 was scored.

Samples ``--n`` responses per prompt with vLLM at the paper's evaluation settings (App. C:
temperature 1.0, top-p 0.8, 8192 tokens, 8 samples) from the prompt sets rendered by
``scripts/miles/prepare_eopd_math_prompts.py`` (already carrying the App. C instruction), then
extracts and grades every answer with the Qwen2.5-Math evaluation harness
(``parser.extract_answer`` + ``grader.math_equal``, 3 s per comparison) and reports Avg@n /
Pass@n per set next to the paper's OPD row. Writes ``samples.jsonl``, ``results.json`` and
``summary.md`` under ``--output``.

    git clone https://github.com/QwenLM/Qwen2.5-Math /tmp/qwen25-math
    python scripts/eopd/qwen25_math_harness_eval.py \\
        --model /weka/.../runs/eopd-opd-qwen3-4b-base-dapo14k-v3/hf-219 \\
        --data-dir /weka/.../miles-opd/data/eopd-math-v1 --harness /tmp/qwen25-math \\
        --paper-student qwen3-4b-base --output /weka/.../qwen25math_eval/arm2-opd-hf-219
"""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from open_instruct import logger_utils, qwen25_math_harness

logger = logger_utils.setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF checkpoint directory (tokenizer alongside)")
    parser.add_argument(
        "--data-dir", required=True, help="directory with <set>.jsonl from prepare_eopd_math_prompts.py"
    )
    parser.add_argument("--sets", default=",".join(qwen25_math_harness.DEFAULT_SETS))
    parser.add_argument("--harness", required=True, help="Qwen2.5-Math checkout")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--grade-timeout", type=float, default=3.0)
    parser.add_argument("--grade-workers", type=int, default=8)
    parser.add_argument("--paper-student", choices=sorted(qwen25_math_harness.PAPER_OPD), default=None)
    parser.add_argument("--olympiadbench-source", default="math-ai/olympiadbench")
    parser.add_argument("--olympiadbench-revision", default="4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84")
    parser.add_argument("--limit", type=int, default=None, help="prompts per set (smoke runs)")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def stop_token_ids(model: Path) -> list[int]:
    config = json.loads((model / "generation_config.json").read_text())
    eos = config.get("eos_token_id")
    return list(eos) if isinstance(eos, list) else [eos]


def sample(args: argparse.Namespace, sets: dict[str, list[dict]], output: Path) -> dict[str, list[dict]]:
    import vllm  # noqa: PLC0415  # keep the module importable without a GPU stack

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = {
        name: [tokenizer(r["input"], add_special_tokens=False)["input_ids"] for r in rows]
        for name, rows in sets.items()
    }
    max_prompt = max(len(ids) for per_set in prompt_ids.values() for ids in per_set)
    stops = stop_token_ids(Path(args.model))
    logger.info("stop token ids %s, longest prompt %d tokens", stops, max_prompt)
    llm = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_prompt + args.max_tokens,
        enable_prefix_caching=True,
    )
    sampling = vllm.SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=stops,
    )
    responses: dict[str, list[dict]] = {}
    with open(output / "samples.jsonl", "w") as f:
        for name, rows in sets.items():
            started = time.time()
            outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids[name]], sampling)
            responses[name] = []
            for index, (row, result) in enumerate(zip(rows, outputs)):
                for k, completion in enumerate(result.outputs):
                    record = {
                        "set": name,
                        "prompt_index": index,
                        "sample_index": k,
                        "sample_id": f"{name}:{index}:{k}",
                        "source_row": row["metadata"].get("source_row"),
                        "text": completion.text,
                        "response_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "label": row["label"],
                    }
                    responses[name].append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("%s: %d prompts x %d samples in %.0f s", name, len(rows), args.n, time.time() - started)
    return responses


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    harness = qwen25_math_harness.attach_harness(Path(args.harness))
    _, strip_string, _ = harness
    names = [s.strip() for s in args.sets.split(",") if s.strip()]
    sets = {
        name: qwen25_math_harness.load_prompts(Path(args.data_dir) / f"{name}.jsonl")[: args.limit] for name in names
    }
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    responses = sample(args, sets, output)

    results, timeouts = {}, {}
    for name, rows in sets.items():
        data_name = qwen25_math_harness.HARNESS_DATA_NAME[name]
        first_answers = None
        if name == "olympiadbench":
            first_answers = qwen25_math_harness.olympiadbench_first_answers(
                rows, args.olympiadbench_source, args.olympiadbench_revision
            )
            logger.info("olympiadbench ground truth from %s", "source dataset" if first_answers else "rendered label")
        for record in responses[name]:
            raw = first_answers[record["source_row"]] if first_answers else record["label"]
            record["ground_truth"] = qwen25_math_harness.normalize_ground_truth(raw, data_name, strip_string)
        graded, timeouts[name] = qwen25_math_harness.grade(
            responses[name], data_name, harness, timeout=args.grade_timeout, workers=args.grade_workers
        )
        results[name] = qwen25_math_harness.aggregate(graded) | {"grader_timeouts": timeouts[name]}
        logger.info(
            "%s: Avg@%d %.2f Pass@%d %.2f", name, args.n, results[name]["avg_at_n"], args.n, results[name]["pass_at_n"]
        )

    with open(output / "graded.jsonl", "w") as f:
        for name in sets:
            for record in responses[name]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    paper = qwen25_math_harness.PAPER_OPD.get(args.paper_student) if args.paper_student else None
    table = qwen25_math_harness.summary_table(results, paper)
    (output / "results.json").write_text(
        json.dumps({"results": results, "paper_opd": paper, "config": vars(args)}, indent=2) + "\n"
    )
    (output / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
