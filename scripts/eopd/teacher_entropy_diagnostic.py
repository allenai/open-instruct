"""Teacher-entropy diagnostic for on-policy distillation (validation program step 5).

Samples student rollouts for a set of rendered prompts (Miles ``data.prompt_data`` JSONL with
an ``input`` field, e.g. the EOPD prompts from ``scripts/miles/prepare_eopd_math_prompts.py``),
then scores every response token with the teacher and records, per token, the exact teacher
entropy, the top-k mass, the renormalized top-k entropy proxy used by EOPD (arXiv 2603.07079),
and the teacher log-probability / rank / top-k membership of the token the student sampled.
Writes ``rollouts.jsonl``, ``token_stats.pt`` and ``summary.json`` under ``--output``.

Two GPU phases run as separate processes so vLLM's memory is released before the teacher
loads; ``all`` chains them:

    python scripts/eopd/teacher_entropy_diagnostic.py all \\
        --student Qwen/Qwen3-4B-Base --teacher Qwen/Qwen3-8B \\
        --prompts /weka/.../eopd-math-v1/dapo_math_14k.jsonl --num-prompts 256 \\
        --output /weka/.../teacher_entropy/qwen3-8b_on_qwen3-4b-base

The student and teacher must share a tokenizer (checked); the teacher scores the student's
exact token sequence, as OPD does.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import logger_utils, teacher_entropy

logger = logger_utils.setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("generate", "score", "all"))
    parser.add_argument("--student", required=True)
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-revision", default=None)
    parser.add_argument("--prompts", required=True, help="JSONL with an `input` field (rendered prompt text)")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--score-chunk", type=int, default=1024, help="response positions per softmax chunk")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def load_prompts(path: Path, num_prompts: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if num_prompts < len(rows):
        rows = random.Random(seed).sample(rows, num_prompts)
    return [{"prompt": row["input"], "metadata": row.get("metadata", {})} for row in rows]


def generate(args: argparse.Namespace, output: Path) -> None:
    # vLLM is only needed in this phase; importing it at module scope would make `score` need it too.
    import vllm  # noqa: PLC0415

    prompts = load_prompts(Path(args.prompts), args.num_prompts, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student, revision=args.student_revision)
    prompt_ids = [tokenizer(p["prompt"], add_special_tokens=False)["input_ids"] for p in prompts]
    max_prompt = max(len(ids) for ids in prompt_ids)
    llm = vllm.LLM(
        model=args.student,
        revision=args.student_revision,
        tokenizer=args.student,
        tokenizer_revision=args.student_revision,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_prompt + args.max_tokens,
        enable_prefix_caching=True,
    )
    sampling = vllm.SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling)
    with open(output / "rollouts.jsonl", "w") as f:
        for index, (prompt, ids, result) in enumerate(zip(prompts, prompt_ids, outputs)):
            for completion in result.outputs:
                f.write(
                    json.dumps(
                        {
                            "prompt_index": index,
                            "prompt_token_ids": ids,
                            "response_token_ids": list(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            "metadata": prompt["metadata"],
                        }
                    )
                    + "\n"
                )
    logger.info(f"Wrote {sum(len(r.outputs) for r in outputs)} rollouts for {len(prompts)} prompts")


def check_shared_tokenizer(args: argparse.Namespace) -> None:
    student = AutoTokenizer.from_pretrained(args.student, revision=args.student_revision)
    teacher = AutoTokenizer.from_pretrained(args.teacher, revision=args.teacher_revision)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("student and teacher tokenizers differ; the teacher cannot score student tokens")


@torch.no_grad()
def score(args: argparse.Namespace, output: Path) -> None:
    check_shared_tokenizer(args)
    rollouts = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, revision=args.teacher_revision, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    chunks, per_rollout, positions = [], [], []
    for rollout in rollouts:
        response = rollout["response_token_ids"]
        if not response:
            continue
        ids = torch.tensor([rollout["prompt_token_ids"] + response], device="cuda")
        prompt_len = len(rollout["prompt_token_ids"])
        logits = model(input_ids=ids).logits[0, prompt_len - 1 : -1]  # predicts each response token
        sampled = torch.tensor(response, device="cuda")
        rollout_chunks = []
        for start in range(0, logits.shape[0], args.score_chunk):
            stop = start + args.score_chunk
            rollout_chunks.append(teacher_entropy.token_statistics(logits[start:stop], sampled[start:stop], args.k))
        stats = teacher_entropy.concatenate(rollout_chunks)
        chunks.append(stats)
        positions.append(torch.arange(len(response)))
        per_rollout.append(
            {
                "prompt_index": rollout["prompt_index"],
                "response_tokens": len(response),
                "finish_reason": rollout["finish_reason"],
                "entropy_mean": float(stats["entropy"].mean()),
                "frac_entropy_gt_tau": float((stats["entropy"] > args.tau).float().mean()),
                "frac_sampled_outside_topk": float((~stats["sampled_in_topk"]).float().mean()),
                "teacher_logprob_mean": float(stats["sampled_logprob"].mean()),
            }
        )
    stats = teacher_entropy.concatenate(chunks)
    stats["position"] = torch.cat(positions) if positions else torch.empty(0, dtype=torch.long)
    torch.save(stats, output / "token_stats.pt")
    summary = {
        "student": args.student,
        "student_revision": args.student_revision,
        "teacher": args.teacher,
        "teacher_revision": args.teacher_revision,
        "prompts": args.prompts,
        "rollouts": len(per_rollout),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
        "frac_truncated": sum(r["finish_reason"] == "length" for r in per_rollout) / max(len(per_rollout), 1),
        **teacher_entropy.summarize(stats, args.tau, args.k),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "per_rollout.json").write_text(json.dumps(per_rollout, indent=1) + "\n")
    logger.info(json.dumps({k: v for k, v in summary.items() if k != "entropy_histogram"}, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        base = [a for a in (sys.argv[1:] if argv is None else argv) if a != "all"]
        for phase in ("generate", "score"):
            subprocess.run([sys.executable, __file__, phase, *base], check=True)
        return
    if args.phase == "generate":
        generate(args, output)
    else:
        score(args, output)


if __name__ == "__main__":
    main()
