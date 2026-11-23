"""Render extra generative math evals in the R1 base-model template used for training and in-loop evals.

- minerva_math: the full Hendrycks MATH test set (5,000 problems, 7 subjects; EleutherAI/hendrycks_math), the set OLMES
  calls minerva_math. The label is the last \\boxed{} / \\fbox{} answer of the reference solution.
- minerva_ocw: the 272 OCW-courses problems from the Minerva paper (math-ai/minervamath), the set RL-zero papers call
  Minerva Math. Each problem is repeated --ocw-repeats times.
- aime2026: AIME 2026 I + II (MathArena/aime_2026, 30 problems), each repeated --aime-repeats times so a run with
  n_samples_per_eval_prompt = 1 reports avg@repeats.

Prompts over --max-prompt-tokens are dropped (counted in the manifest), as for training data.
Writes eval-<name>.jsonl, verifiers.json and manifest.json into OUTPUT, for [data] eval_prompt_data / reward_config.

Usage: python prepare_extra_evals.py OUTPUT --tokenizer /path/to/olmicro/hf/checkpoint
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_olmicro_math as base  # noqa: E402

MATH_FULL = ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3")
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")
OCW = ("math-ai/minervamath", "ee46ddc498933b1977577953250ca5c66be64f96", "test.jsonl")
AIME = ("MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112", "data/train-00000-of-00001.parquet")


def last_boxed(solution):
    """Content of the last \\boxed{...} or \\fbox{...} in a MATH reference solution (brace-matched)."""
    start = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if start < 0:
        raise ValueError("no boxed answer")
    i = solution.index("{", start)
    depth = 0
    for j in range(i, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        if depth == 0:
            return solution[i + 1 : j].strip()
    raise ValueError("unbalanced boxed answer")


def records(name, repo, revision, items, repeats, tokenizer, max_prompt_tokens):
    out, dropped = [], 0
    for index, (problem, label, extra) in enumerate(items):
        prompt = base.TEMPLATES["r1"][0](problem.strip())
        if len(tokenizer.encode(prompt, add_special_tokens=False).ids) > max_prompt_tokens:
            dropped += 1
            continue
        for copy in range(repeats):
            sample_id = f"{repo}:{revision}:{index}" + (f"#{copy}" if repeats > 1 else "")
            out.append({
                "input": prompt,
                "label": label,
                "metadata": {
                    "prepared_sample_id": sample_id, "source_dataset": repo, "source_revision": revision, "source_row": index,
                    "query": problem, "verifiers": [{"name": "math", "target": label, "weight": 1.0}], **extra,
                },
            })
    return name, out, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--aime-repeats", type=int, default=32)
    parser.add_argument("--ocw-repeats", type=int, default=4)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    args.output.mkdir(parents=True, exist_ok=False)

    math_items = []
    for subject in MATH_SUBJECTS:
        path = hf_hub_download(MATH_FULL[0], f"{subject}/test-00000-of-00001.parquet", revision=MATH_FULL[1], repo_type="dataset")
        for row in pq.read_table(path).to_pylist():
            math_items.append((row["problem"], last_boxed(row["solution"]), {"subject": subject, "level": row["level"], "reference_solution": row["solution"]}))
    ocw_path = hf_hub_download(OCW[0], OCW[2], revision=OCW[1], repo_type="dataset")
    ocw_items = [(row["question"], str(row["answer"]).strip(), {}) for row in map(json.loads, Path(ocw_path).read_text().splitlines()) if row]
    aime_path = hf_hub_download(AIME[0], AIME[2], revision=AIME[1], repo_type="dataset")
    aime_items = [(row["problem"], str(row["answer"]).strip(), {"problem_idx": str(row["problem_idx"])}) for row in pq.read_table(aime_path).to_pylist()]

    sets = [
        records("minerva_math", *MATH_FULL, math_items, 1, tokenizer, args.max_prompt_tokens),
        records("minerva_ocw", OCW[0], OCW[1], ocw_items, args.ocw_repeats, tokenizer, args.max_prompt_tokens),
        records("aime2026", AIME[0], AIME[1], aime_items, args.aime_repeats, tokenizer, args.max_prompt_tokens),
    ]
    manifest = {"template": "r1", "required_miles_stop": base.TEMPLATES["r1"][1], "sets": {}}
    for name, rows, dropped in sets:
        path = args.output / f"eval-{name}.jsonl"
        base.write(path, rows)
        manifest["sets"][name] = {"rows": len(rows), "dropped_over_length": dropped, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest["repeats"] = {"aime2026": args.aime_repeats, "minerva_ocw": args.ocw_repeats, "minerva_math": 1}
    manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "verifiers.json").write_text(json.dumps({"math": base.REGISTRY["math"]}, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
