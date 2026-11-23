"""Render Dolci RL-Zero math (train) plus MATH-500 and GSM8K Platinum (eval) for a base Olmicro model.

Prompts are plain text with no chat control tokens, matching the validated GSM8K base recipe
(scripts/data/miles/prepare_hybrid_gsm8k.py in Open-Instruct), but ask for a \\boxed{} answer so
open_instruct.ground_truth_utils.MathVerifier can extract it. The OLMo 3 RL-Zero "Answer: X" line is
not parsed by MathVerifier, so it is deliberately not used here.

Writes train.jsonl, eval-math500.jsonl, eval-gsm8k_platinum.jsonl, eval-smoke.jsonl, verifiers.json
and manifest.json into OUTPUT, for [data] prompt_data / eval_prompt_data / reward_config.

Usage: python prepare_olmicro_math.py OUTPUT --tokenizer /path/to/olmicro/hf/checkpoint
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TRAIN = ("allenai/Dolci-RL-Zero-Math-7B", "93b5e498577804a93fb11bfe8821428f8535f2d8", "data/train-00000-of-00001.parquet")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl")
GSM8K_PLATINUM = ("mnoukhov/gsm8k-platinum-openinstruct", "409e774196520a279cfc6b08cc4924b1cfd4927f", "data/test-00000-of-00001.parquet")
REGISTRY = {
    "math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"},
    "gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"},
}
INSTRUCTION = "Solve the following problem step by step, and put your final answer within \\boxed{}."
# Keep in sync with tools/format_probe.py, which measured these templates on the base model.
# Each entry: (renderer, the [miles] stop setting the run must use with it).
TEMPLATES = {
    "qa_boxed": (lambda q: f"{INSTRUCTION}\n\nQuestion: {q}\nAnswer:", {"rollout_stop": ["\nQuestion:"]}),
    "qa_boxed_think": (
        lambda q: f"{INSTRUCTION}\n\nQuestion: {q}\nAnswer: Let's think step by step.",
        {"rollout_stop": ["\nQuestion:"]},
    ),
    "problem_solution": (
        lambda q: f"Problem:\n{q}\n\nPut your final answer within \\boxed{{}}.\n\nSolution:",
        {"rollout_stop": ["\nProblem:"]},
    ),
    "chatml": (
        lambda q: f"<|im_start|>user\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n",
        {"rollout_stop_token_ids": [100264, 100265]},
    ),
    "r1": (
        lambda q: (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process step by step and then provides the User with "
            "the answer. The final answer is enclosed within \\boxed{}.\n"
            f"User: {q}\nAssistant:"
        ),
        {"rollout_stop": ["\nUser:"]},
    ),
}
TEMPLATE = "qa_boxed"


def render(problem):
    return TEMPLATES[TEMPLATE][0](problem.strip())


def fetch(spec):
    repo, revision, filename = spec
    return Path(hf_hub_download(repo, filename, revision=revision, repo_type="dataset"))


def read_rows(spec):
    path = fetch(spec)
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def single(target):
    if isinstance(target, list):
        if len(target) != 1:
            raise ValueError(f"Expected one ground truth, got {len(target)}")
        target = target[0]
    return str(target).strip()


def build(name, spec, rows, problem_of, target_of, verifier, tokenizer, max_prompt_tokens):
    repo, revision, _ = spec
    records, dropped = [], 0
    for index, row in enumerate(rows):
        problem, target = problem_of(row), single(target_of(row))
        prompt = render(problem)
        if len(tokenizer.encode(prompt, add_special_tokens=False).ids) > max_prompt_tokens:
            dropped += 1
            continue
        records.append(
            {
                "input": prompt,
                "label": target,
                "metadata": {
                    "prepared_sample_id": f"{repo}:{revision}:{index}",
                    "source_dataset": repo,
                    "source_revision": revision,
                    "source_row": index,
                    "query": problem,
                    "verifiers": [{"name": verifier, "target": target, "weight": 1.0}],
                },
            }
        )
    return name, records, dropped


def write(path, records):
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"path": path.name, "records": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def user_content(row):
    messages = row["messages"] if len(row["messages"]) == 1 else row["messages"][:-1]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Expected one user message")
    return messages[0]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", required=True, help="HF checkpoint directory containing tokenizer.json")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--smoke-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--template", choices=sorted(TEMPLATES), default="qa_boxed")
    args = parser.parse_args()
    global TEMPLATE
    TEMPLATE = args.template
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    args.output.mkdir(parents=True, exist_ok=False)

    sets = [
        build("train", TRAIN, read_rows(TRAIN), lambda r: r["prompt"], lambda r: r["ground_truth"], "math", tokenizer, args.max_prompt_tokens),
        build("math500", MATH500, read_rows(MATH500), lambda r: r["problem"], lambda r: r["answer"], "math", tokenizer, args.max_prompt_tokens),
        build("gsm8k_platinum", GSM8K_PLATINUM, read_rows(GSM8K_PLATINUM), user_content, lambda r: r["ground_truth"], "gsm8k", tokenizer, args.max_prompt_tokens),
    ]
    artifacts, sources = {}, {}
    for name, records, dropped in sets:
        filename = "train.jsonl" if name == "train" else f"eval-{name}.jsonl"
        artifacts[name] = write(args.output / filename, records)
        sources[name] = {"dropped_over_length": dropped}
    # Small mixed eval for the lifecycle smoke: the smoke copies the first rows of this file.
    math500 = next(records for name, records, _ in sets if name == "math500")
    gsm8k = next(records for name, records, _ in sets if name == "gsm8k_platinum")
    rng = random.Random(args.seed)
    smoke = rng.sample(math500, args.smoke_rows // 2) + rng.sample(gsm8k, args.smoke_rows - args.smoke_rows // 2)
    artifacts["smoke"] = write(args.output / "eval-smoke.jsonl", smoke)

    (args.output / "verifiers.json").write_text(json.dumps(REGISTRY, indent=2) + "\n")
    manifest = {
        "prompt_format": f"olmicro_base_{TEMPLATE}_v1",
        "template_name": TEMPLATE,
        "template": render("{problem}"),
        "required_miles_stop": TEMPLATES[TEMPLATE][1],
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": hashlib.sha256((Path(args.tokenizer) / "tokenizer.json").read_bytes()).hexdigest(),
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "inputs": {"train": TRAIN, "math500": MATH500, "gsm8k_platinum": GSM8K_PLATINUM},
        "sources": sources,
        "artifacts": artifacts,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
