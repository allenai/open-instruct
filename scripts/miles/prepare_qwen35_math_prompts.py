"""Render the Qwen3.5 math OPD prompts for Miles from the Open Instruct campaign data.

The Open Instruct runs (scripts/general_agent/terminal/rl/qwen35_*_math_*.sh) train on a
fixed DAPO split and evaluate on the DAPO holdout, AIME 2025, BRUMO 2025 and MATH-500 with
``--chat_template qwen_instruct_user_boxed_math`` and the ``math`` verifier. This script writes
the same prompts, already rendered with that template, as Miles ``data.prompt_data`` /
``data.eval_prompt_data`` JSONL plus the ``verifiers.json`` registry for ``data.reward_config``.

Example (the DAPO split and MATH-500 files are the campaign's Beaker datasets):

    python scripts/miles/prepare_qwen35_math_prompts.py \\
        --tokenizer Qwen/Qwen3.5-2B --train dapo/train.jsonl \\
        --eval dapo_math_holdout=dapo/eval.jsonl --eval math_500=math500/eval.jsonl \\
        --eval math_aime_2025=hf:mnoukhov/aime_2025_openinstruct@c968739ed3b8d0f1f35c1612525ef1484f08b267 \\
        --eval math_brumo_2025=hf:mnoukhov/brumo_2025_openinstruct@997f311ed7d30147c32ebd85f342b285b78b0b3b \\
        --output /weka/.../miles-opd/data/qwen35-math-v1
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import datasets
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES

logger = logger_utils.setup_logger(__name__)

TEMPLATE_NAME = "qwen_instruct_user_boxed_math"
VERIFIERS = {"math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"}}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_rows(location):
    """Read ``path.jsonl`` or ``hf:repo@revision`` into (rows, provenance)."""
    if location.startswith("hf:"):
        repo, _, revision = location[3:].partition("@")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit(f"{location}: Hugging Face sources need an immutable 40-character revision")
        dataset = datasets.load_dataset(repo, revision=revision, split="train")
        return [dict(row) for row in dataset], {"source": repo, "revision": revision, "split": "train"}
    path = Path(location)
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return rows, {"source": str(path.resolve()), "sha256": _sha(raw)}


def render(name, rows, tokenizer, template, provenance):
    rendered = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise SystemExit(f"{name} row {index}: expected messages ending in a user turn")
        target = row.get("ground_truth")
        if not isinstance(target, str) or not target.strip():
            raise SystemExit(f"{name} row {index}: expected a nonempty ground_truth string")
        prompt = tokenizer.apply_chat_template(
            messages, chat_template=template, tokenize=False, add_generation_prompt=True
        )
        rendered.append(
            {
                "input": prompt,
                "label": target,
                "metadata": {
                    "prepared_sample_id": f"{name}:{index}",
                    "source_dataset": provenance["source"],
                    "source_row": index,
                    "dataset": row.get("dataset"),
                    "query": messages[-1]["content"],
                    "opd_messages": messages,
                    "verifiers": [{"name": "math", "target": target, "weight": 1.0}],
                },
            }
        )
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="Learner tokenizer (HF id or local checkpoint)")
    parser.add_argument("--tokenizer-revision", default=None, help="Immutable revision for an HF tokenizer id")
    parser.add_argument("--train", required=True, help="Training prompts: JSONL path or hf:repo@revision")
    parser.add_argument("--eval", action="append", default=[], metavar="NAME=SOURCE", help="Held-out set; repeatable")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Fail if a rendered prompt exceeds this")
    parser.add_argument(
        "--note", action="append", default=[], metavar="KEY=VALUE", help="Provenance recorded in manifest.json"
    )
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; choose a fresh output directory")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    template = CHAT_TEMPLATES[TEMPLATE_NAME]
    sets = {"train": args.train}
    for item in args.eval:
        name, _, location = item.partition("=")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name == "train" or not location:
            raise SystemExit(f"--eval expects NAME=SOURCE with a safe, non-train name; got {item!r}")
        if name in sets:
            raise SystemExit(f"Duplicate evaluation set {name!r}")
        sets[name] = location

    output.mkdir(parents=True)
    manifest = {"template": TEMPLATE_NAME, "template_sha256": _sha(template.encode()), "tokenizer": args.tokenizer}
    manifest["tokenizer_revision"] = args.tokenizer_revision
    manifest["notes"] = dict(note.partition("=")[::2] for note in args.note)
    manifest["sets"] = {}
    rendered_sets = {}
    for name, location in sets.items():
        rows, provenance = load_rows(location)
        rendered_sets[name] = (render(name, rows, tokenizer, template, provenance), provenance)
    # Miles refuses training prompts that also appear held out. The DAPO source repeats a
    # few problems, so drop the training copies and keep every evaluation set intact.
    held_out = {row["input"] for name, (rows, _) in rendered_sets.items() if name != "train" for row in rows}
    train_rows, provenance = rendered_sets["train"]
    dropped = [row["metadata"]["source_row"] for row in train_rows if row["input"] in held_out]
    rendered_sets["train"] = ([row for row in train_rows if row["input"] not in held_out], provenance)
    manifest["train_rows_dropped_for_held_out_overlap"] = dropped
    if dropped:
        logger.info("Dropped %d training prompts that also appear in an evaluation set: %s", len(dropped), dropped)
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
            "sha256": _sha(raw),
            "max_prompt_tokens": max(lengths),
        }
        logger.info("%s: %d prompts, longest %d tokens", name, len(rendered), max(lengths))
    (output / "chat_template.jinja").write_text(template)
    (output / "verifiers.json").write_text(json.dumps(VERIFIERS, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
