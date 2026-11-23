"""Compare base-model prompt templates for RL-zero math with native SGLang, at the RL sampling settings.

For each template, samples --samples completions per prompt (temperature 1.0, top_p --top-p (default 1.0), --max-new
tokens) for a fixed set of training and MATH-500 prompts, writes raw responses; score them with score_format_probe.py.

Run inside the MILES image from /opt/core-rl:
  CUDA_VISIBLE_DEVICES=0 python format_probe.py --model M --data D --out O --templates qa_boxed r1
"""

import argparse
import json
import random
import time
from pathlib import Path

INSTRUCTION = "Solve the following problem step by step, and put your final answer within \\boxed{}."
TEMPLATES = {
    # Current v1 data format.
    "qa_boxed": (lambda q: f"{INSTRUCTION}\n\nQuestion: {q}\nAnswer:", {"stop": ["\nQuestion:"]}),
    "qa_boxed_think": (
        lambda q: f"{INSTRUCTION}\n\nQuestion: {q}\nAnswer: Let's think step by step.",
        {"stop": ["\nQuestion:"]},
    ),
    "problem_solution": (
        lambda q: f"Problem:\n{q}\n\nPut your final answer within \\boxed{{}}.\n\nSolution:",
        {"stop": ["\nProblem:"]},
    ),
    "chatml": (
        lambda q: f"<|im_start|>user\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n",
        {"stop_token_ids": [100264, 100265]},
    ),
    "r1": (
        lambda q: (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process step by step and then provides the User with "
            "the answer. The final answer is enclosed within \\boxed{}.\n"
            f"User: {q}\nAssistant:"
        ),
        {"stop": ["\nUser:"]},
    ),
}


def load(path, count, seed):
    """count < 0 keeps every row in file order (pass-rate sweeps); otherwise a seeded sample."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if count >= 0:
        rows = random.Random(seed).sample(rows, min(count, len(rows)))
    return [
        {"id": row["metadata"]["prepared_sample_id"], "query": row["metadata"]["query"], "label": row["label"], "input": row["input"]}
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--templates", nargs="+", required=True, choices=sorted(TEMPLATES))
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-math500", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4608)
    parser.add_argument("--from-input", action="store_true", help="Use each row's prepared input verbatim (sweeps)")
    parser.add_argument("--shard", type=int, nargs="+", default=[0], help="Shard indices to run, in order, in one engine")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eval-files", nargs="+", default=[], metavar="NAME=PATH",
                        help="Sample every row of these prepared eval files instead of train/MATH-500 (use with --from-input)")
    args = parser.parse_args()

    import sglang
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.eval_files:
        everything = [{"source": name, **r} for name, path in (spec.split("=", 1) for spec in args.eval_files) for r in load(path, -1, args.seed)]
    else:
        everything = [{"source": "train", **r} for r in load(args.data / "train.jsonl", args.n_train, args.seed)]
        everything += [{"source": "math500", **r} for r in load(args.data / "eval-math500.jsonl", args.n_math500, args.seed)]

    def output_path(name, shard):
        suffix = f".shard{shard:02d}of{args.num_shards:02d}" if args.num_shards > 1 else ""
        return args.out / f"{name}{suffix}.raw.json"

    # Preemption tolerance: shards already written (atomically, below) are skipped on a restart.
    todo = [(s, n) for s in args.shard for n in args.templates if not output_path(n, s).exists()]
    if not todo:
        print("FORMAT_PROBE_NOTHING_TO_DO", flush=True)
        return
    engine = sglang.Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        context_length=args.context_length,
        attention_backend="triton",
        disable_cuda_graph=False,
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=args.concurrency,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        max_running_requests=args.concurrency,
        mem_fraction_static=0.6,
        random_seed=args.seed,
        linear_attn_backend="triton",
        linear_attn_decode_backend="triton",
        linear_attn_prefill_backend="triton",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for shard, name in todo:
        prompts = everything[shard :: args.num_shards]
        render, stops = TEMPLATES[name]
        started = time.monotonic()
        texts = [p["input"] if args.from_input else render(p["query"]) for p in prompts]
        ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        flat = [i for i in ids for _ in range(args.samples)]
        params = {"temperature": 1.0, "top_p": args.top_p, "max_new_tokens": args.max_new, "no_stop_trim": True, **stops}
        outputs = engine.generate(input_ids=flat, sampling_params=params)
        raw = []
        for p, (prompt, text) in enumerate(zip(prompts, texts)):
            group = outputs[p * args.samples : (p + 1) * args.samples]
            raw.append({"source": prompt["source"], "id": prompt["id"], "query": prompt["query"], "label": prompt["label"], "prompt": text, "responses": [
                {"text": o["text"], "finish": o["meta_info"]["finish_reason"], "tokens": o["meta_info"]["completion_tokens"]}
                for o in group
            ]})
        # Scoring runs separately (score_format_probe.py): MathVerifier needs ANTLR 4.11, which the image keeps
        # in a child-only directory so the SGLang/rollout process can keep ANTLR 4.9.
        meta = {"template": name, "example_prompt": render("{problem}"), "stops": stops, "seconds": round(time.monotonic() - started, 1),
                "from_input": args.from_input, "shard": shard, "num_shards": args.num_shards, "samples": args.samples, "max_new": args.max_new, "top_p": args.top_p}
        target = output_path(name, shard)
        partial = target.with_suffix(".partial")
        partial.write_text(json.dumps({"meta": meta, "prompts": raw}))
        partial.rename(target)
        print("FORMAT_PROBE_GENERATED " + json.dumps(meta), flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
