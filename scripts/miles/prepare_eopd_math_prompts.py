"""Render the EOPD (arXiv 2603.07079) math prompts for the Miles OPD replication.

The paper trains Qwen3-Base students from a Qwen3-8B teacher on the MATH training set
(7.5k problems, DigitalLearningGmbH/MATH-lighteval) or DAPO-Math-14k (the ``en`` subset of
open-r1/DAPO-Math-17k-Processed, 14,116 problems) and evaluates with the Qwen2.5-Math harness
on MATH-500, AMC23, Minerva Math, OlympiadBench, AIME24 and AIME25. Appendix A renders every
problem as one user turn followed by "Please reason step by step, and put your final answer
within \\boxed{}." through the Qwen3 chat template with thinking disabled (the assistant turn
starts with an empty ``<think></think>`` block; the teacher runs in non-thinking mode). The
repository's verl preprocessing scripts use the wording "Let's think step by step and output
the final answer within \\boxed{}." instead; ``--instruction`` selects either. This script
writes the paper's prompts as Miles
``data.prompt_data`` / ``data.eval_prompt_data`` JSONL (one file per set) plus the
``verifiers.json`` registry for ``data.reward_config``, mirroring
``scripts/miles/prepare_qwen35_math_prompts.py``.

Every source is pinned to an immutable Hugging Face revision. Only the ``datasets`` and
``transformers`` packages are needed, so it runs in a slim CPU image:

    python scripts/miles/prepare_eopd_math_prompts.py \\
        --tokenizer Qwen/Qwen3-4B-Base \\
        --tokenizer-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \\
        --output /weka/.../miles-opd/data/eopd-math-v1

Qwen3-1.7B-Base and Qwen3-4B-Base share the Qwen3 tokenizer and chat template, so one
output directory serves both paper settings.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import datasets
from transformers import AutoTokenizer

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
VERIFIERS = {"math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"}}
# name -> (repo, revision, config, split, question field, answer field). Answer fields may be
# prefixed with ``boxed:`` (take the last \\boxed{} of a worked solution) or ``list:`` (a
# Python-literal list of answers; joined with ", " when there are several).
SETS = {
    "math_train": (
        "DigitalLearningGmbH/MATH-lighteval",
        "0530c78699ea5e8eb5530600900e1f328b48acad",
        "default",
        "train",
        "problem",
        "boxed:solution",
    ),
    "dapo_math_14k": (
        "open-r1/DAPO-Math-17k-Processed",
        "31dd309567e3da778038cc87d868b6097a3ccf68",
        "en",
        "train",
        "prompt",
        "solution",
    ),
    "math_500": (
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        None,
        "test",
        "problem",
        "answer",
    ),
    "aime24": (
        "Maxwell-Jia/AIME_2024",
        "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        None,
        "train",
        "Problem",
        "Answer",
    ),
    "aime25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", None, "test", "problem", "answer"),
    "amc23": ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057", None, "test", "question", "answer"),
    "minerva_math": (
        "math-ai/minervamath",
        "ee46ddc498933b1977577953250ca5c66be64f96",
        None,
        "test",
        "question",
        "answer",
    ),
    "olympiadbench": (
        "math-ai/olympiadbench",
        "4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84",
        None,
        "test",
        "question",
        "list:final_answer",
    ),
}
TRAIN_SETS = ("math_train", "dapo_math_14k")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def last_boxed(solution: str) -> str:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``) with balanced braces."""
    start = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if start < 0:
        raise ValueError("no \\boxed answer")
    index = solution.index("{", start)
    depth = 0
    for end in range(index, len(solution)):
        if solution[end] == "{":
            depth += 1
        elif solution[end] == "}":
            depth -= 1
            if depth == 0:
                return solution[index + 1 : end].strip()
    raise ValueError("unbalanced \\boxed answer")


def answer_of(row: dict, field: str) -> tuple[str, bool]:
    """Return (label, is_multiple_answer)."""
    if field.startswith("boxed:"):
        return last_boxed(str(row[field[6:]])), False
    if field.startswith("list:"):
        value = row[field[5:]]
        answers = ast.literal_eval(value) if isinstance(value, str) else list(value)
        answers = [str(a).strip() for a in answers]
        return ", ".join(answers), len(answers) > 1
    return str(row[field]).strip(), False


def load_rows(name: str) -> tuple[list[dict], dict]:
    repo, revision, config, split, _, _ = SETS[name]
    dataset = datasets.load_dataset(repo, config, revision=revision, split=split)
    provenance = {"source": repo, "revision": revision, "config": config, "split": split}
    return [dict(row) for row in dataset], provenance


def render(
    name: str, rows: list[dict], tokenizer, enable_thinking: bool, instruction: str, provenance: dict
) -> tuple[list[dict], dict]:
    _, _, _, _, question_field, answer_field = SETS[name]
    rendered, skipped, multiple = [], [], 0
    for index, row in enumerate(rows):
        question = str(row[question_field]).strip()
        try:
            target, is_multiple = answer_of(row, answer_field)
        except (ValueError, KeyError, SyntaxError) as error:
            skipped.append({"source_row": index, "reason": str(error)})
            continue
        if not question or not target:
            skipped.append({"source_row": index, "reason": "empty question or answer"})
            continue
        multiple += int(is_multiple)
        messages = [{"role": "user", "content": f"{question} {instruction}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        rendered.append(
            {
                "input": prompt,
                "label": target,
                "metadata": {
                    "prepared_sample_id": f"{name}:{index}",
                    "source_dataset": provenance["source"],
                    "source_row": index,
                    "dataset": name,
                    "query": messages[-1]["content"],
                    "opd_messages": messages,
                    "verifiers": [{"name": "math", "target": target, "weight": 1.0}],
                },
            }
        )
    return rendered, {"skipped": skipped, "multiple_answer_rows": multiple}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="Student tokenizer (HF id or local checkpoint)")
    parser.add_argument("--tokenizer-revision", default=None, help="Immutable revision for an HF tokenizer id")
    parser.add_argument("--sets", default=",".join(SETS), help="Comma-separated subset of: " + ", ".join(SETS))
    parser.add_argument(
        "--enable-thinking",
        choices=("true", "false"),
        default="false",
        help="Qwen3 chat-template thinking switch; the paper uses the teacher in non-thinking mode",
    )
    parser.add_argument("--instruction", default=INSTRUCTION, help="Suffix appended to every problem statement")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Fail if a rendered prompt exceeds this")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; choose a fresh output directory")
    names = [name.strip() for name in args.sets.split(",") if name.strip()]
    unknown = sorted(set(names) - set(SETS))
    if unknown:
        raise SystemExit(f"Unknown sets {unknown}; choose from {sorted(SETS)}")
    enable_thinking = args.enable_thinking == "true"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)

    rendered_sets = {}
    for name in names:
        rows, provenance = load_rows(name)
        rendered, notes = render(name, rows, tokenizer, enable_thinking, args.instruction, provenance)
        rendered_sets[name] = (rendered, {**provenance, **notes, "source_rows": len(rows)})
        print(f"{name}: {len(rendered)} prompts from {len(rows)} rows ({len(notes['skipped'])} skipped)")

    # Miles refuses training prompts that also appear held out.
    held_out = {row["input"] for name, (rows, _) in rendered_sets.items() if name not in TRAIN_SETS for row in rows}
    for name in TRAIN_SETS:
        if name not in rendered_sets:
            continue
        rows, provenance = rendered_sets[name]
        dropped = [row["metadata"]["source_row"] for row in rows if row["input"] in held_out]
        rendered_sets[name] = ([row for row in rows if row["input"] not in held_out], provenance)
        provenance["train_rows_dropped_for_held_out_overlap"] = dropped
        if dropped:
            print(f"{name}: dropped {len(dropped)} prompts that also appear in an evaluation set")

    output.mkdir(parents=True)
    manifest = {
        "paper": "arXiv 2603.07079 (EOPD)",
        "instruction": args.instruction,
        "chat_template": "tokenizer default",
        "enable_thinking": enable_thinking,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "sets": {},
    }
    for name, (rendered, provenance) in rendered_sets.items():
        lengths = [len(tokenizer(row["input"], add_special_tokens=False)["input_ids"]) for row in rendered]
        if max(lengths) > args.max_prompt_tokens:
            raise SystemExit(f"{name}: longest prompt has {max(lengths)} tokens > {args.max_prompt_tokens}")
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rendered).encode()
        (output / f"{name}.jsonl").write_bytes(raw)
        manifest["sets"][name] = {
            **provenance,
            "path": f"{name}.jsonl",
            "records": len(rendered),
            "max_prompt_tokens": max(lengths),
            "sha256": _sha(raw),
        }
    (output / "verifiers.json").write_text(json.dumps(VERIFIERS, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rendered_sets)} sets, verifiers.json and manifest.json to {output}")


if __name__ == "__main__":
    if not re.fullmatch(r"[0-9a-f]{40}", SETS["math_train"][1]):
        raise SystemExit("SETS revisions must be immutable 40-character commits")
    main()
