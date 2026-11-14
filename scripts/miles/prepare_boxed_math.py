"""Prepare GSM8K or Dolci RL Zero math for MILES using the learner's chat template.

Run as ``python -m scripts.miles.prepare_boxed_math --help``. The boxed-answer
instruction belongs to these dataset artifacts, not a trainer prompt wrapper.
"""

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import datasets
from transformers import AutoTokenizer

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SOURCES = {
    "gsm8k": ("ai2-adapt-dev/rlvr_gsm8k_zs", "93ffaae6cd2acb8f821f6d4712651320a889b1b9", "gsm8k"),
    "dolci": ("allenai/Dolci-RL-Zero-Math-7B", "93b5e498577804a93fb11bfe8821428f8535f2d8", "math"),
}
FACTORIES = {
    "gsm8k": "open_instruct.miles.rewards.verifiers.GSM8KVerifier",
    "math": "open_instruct.miles.rewards.math_verifier.MathVerifier",
}
AIME_SOURCE = "allenai/aime-2022-2025"
AIME_REVISION = "73e1eba765ad5847cdb5d1e2e7aaf7b22b585798"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def messages_for(row):
    """Keep instructions and the question, excluding reference assistant solutions."""
    messages = copy.deepcopy(row.get("messages"))
    if messages is None:
        messages = [{"role": "user", "content": row.get("question", row.get("prompt"))}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected nonempty messages")
    prompt = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Expected text messages")
        if message.get("role") == "assistant":
            break
        if message.get("role") not in ("system", "user") or not isinstance(message.get("content"), str):
            raise ValueError("Expected text system/user messages")
        prompt.append(message)
    if not prompt or prompt[-1]["role"] != "user" or not prompt[-1]["content"].strip():
        raise ValueError("Expected a nonempty final user question")
    question = prompt[-1]["content"].rstrip()
    if question.endswith(INSTRUCTION):
        question = question[: -len(INSTRUCTION)].rstrip()
    if not question or INSTRUCTION in question:
        raise ValueError("Empty question or repeated boxed-answer instruction")
    prompt[-1]["content"] = question + "\n\n" + INSTRUCTION
    return prompt, question


def prepare_rows(name, rows, tokenizer, *, seed=17, max_prompt_tokens=2048, held_out=()):
    source, revision, verifier = SOURCES[name]
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("The learner tokenizer must provide a chat template")
    excluded_questions = {" ".join(question.split()) for question in held_out}
    rendered, excluded = [], []
    # Match the named-task GSM8K sampling and final training shuffle.
    indices = random.Random(f"{seed}:{name}").sample(range(len(rows)), len(rows))
    for index in indices:
        row = rows[index]
        messages, question = messages_for(row)
        if " ".join(question.split()) in excluded_questions:
            excluded.append(index)
            continue
        target = row.get("ground_truth")
        if target is None and verifier == "gsm8k":
            target = row.get("answer", "").split("####")[-1]
        if isinstance(target, list) and len(target) == 1:
            target = target[0]
        if not isinstance(target, (str, int, float)) or isinstance(target, bool) or not str(target).strip():
            raise ValueError(f"{name} row {index}: expected a scalar target")
        target = str(target).strip()
        if verifier == "gsm8k":
            target = target.replace(",", "")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count(INSTRUCTION) != 1:
            raise ValueError("Chat template must preserve the instruction exactly once")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens or len(tokens) > max_prompt_tokens:
            raise ValueError(f"{name} row {index}: {len(tokens)} prompt tokens exceeds 1..{max_prompt_tokens}")
        rendered.append(
            {
                "input": text,
                "label": target,
                "metadata": {
                    "prepared_sample_id": f"{name}:train:{index}",
                    "source_dataset": source,
                    "source_revision": revision,
                    "source_row": index,
                    "query": messages[-1]["content"],
                    "rendering_template_sha256": sha(template.encode()),
                    "prompt_token_ids": tokens,
                    "verifiers": [{"name": verifier, "target": target, "weight": 1.0}],
                },
            }
        )
    if not rendered:
        raise ValueError("No training rows remain")
    random.Random(f"{seed}:train").shuffle(rendered)
    return rendered, sorted(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SOURCES)
    parser.add_argument("--tokenizer", required=True, help="Learner HF checkpoint directory or repository")
    parser.add_argument("--tokenizer-revision", help="Required immutable commit for a remote tokenizer")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    if args.max_prompt_tokens < 1 or args.seed < 0:
        parser.error("Token limit must be positive and seed nonnegative")
    if not Path(args.tokenizer).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", args.tokenizer_revision or ""):
        parser.error("Remote tokenizers require an immutable --tokenizer-revision")
    source, revision, verifier = SOURCES[args.dataset]
    rows = datasets.load_dataset(source, revision=revision, split="train")
    held_out = []
    if args.dataset == "dolci":
        aime = datasets.load_dataset(AIME_SOURCE, revision=AIME_REVISION, split="train")
        held_out = [row["problem"] for row in aime if row["year"] in (2024, 2025)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    rendered, excluded = prepare_rows(
        args.dataset, rows, tokenizer, seed=args.seed, max_prompt_tokens=args.max_prompt_tokens, held_out=held_out
    )
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rendered).encode()
    manifest = {
        "source": source,
        "revision": revision,
        "split": "train",
        "source_records": len(rows),
        "records": len(rendered),
        "seed": args.seed,
        "instruction": INSTRUCTION,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "template_sha256": sha(tokenizer.chat_template.encode()),
        "train_sha256": sha(raw),
        "max_prompt_tokens": max(len(row["metadata"]["prompt_token_ids"]) for row in rendered),
        "excluded_aime_overlap_rows": excluded,
        "overlap_check": {
            "source": AIME_SOURCE,
            "revision": AIME_REVISION,
            "years": [2024, 2025],
            "method": "whitespace-normalized exact question",
        }
        if held_out
        else None,
    }
    args.output.mkdir(parents=True)
    (args.output / "train.jsonl").write_bytes(raw)
    (args.output / "verifiers.json").write_text(
        json.dumps({verifier: {"factory": FACTORIES[verifier]}}, indent=2) + "\n"
    )
    (args.output / "chat_template.jinja").write_text(tokenizer.chat_template)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
