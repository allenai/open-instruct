"""Re-render a prepared Olmicro math dataset (e.g. a pass band) in another prompt template, keeping the same rows.

Every record's `input` is rebuilt from `metadata.query` with prepare_olmicro_math.TEMPLATES[--template]. The script
first checks that --source-template reproduces each existing `input` byte-for-byte, so only the template changes.
Prompts must still fit --max-prompt-tokens: rows are never dropped, so the train and eval sets stay identical.

Usage: python rerender_template.py SOURCE OUTPUT --source-template r1 --template chatml --tokenizer /path/to/hf/checkpoint
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer

import prepare_olmicro_math as base

FILES = ["train.jsonl", "eval-math500.jsonl", "eval-gsm8k_platinum.jsonl", "eval-smoke.jsonl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-template", required=True, choices=sorted(base.TEMPLATES))
    parser.add_argument("--template", required=True, choices=sorted(base.TEMPLATES))
    parser.add_argument("--tokenizer", required=True, help="HF checkpoint directory containing tokenizer.json")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    old, new = base.TEMPLATES[args.source_template][0], base.TEMPLATES[args.template][0]
    args.output.mkdir(parents=True, exist_ok=False)

    artifacts, longest = {}, {}
    for name in FILES:
        records = [json.loads(line) for line in (args.source / name).read_text().splitlines() if line.strip()]
        for record in records:
            query = record["metadata"]["query"].strip()
            if old(query) != record["input"]:
                raise SystemExit(f"{name}: {record['metadata']['prepared_sample_id']} is not {args.source_template}-rendered")
            record["input"] = new(query)
        lengths = [len(tokenizer.encode(r["input"], add_special_tokens=False).ids) for r in records]
        if max(lengths) > args.max_prompt_tokens:
            raise SystemExit(f"{name}: {sum(n > args.max_prompt_tokens for n in lengths)} prompts exceed {args.max_prompt_tokens} tokens")
        artifacts[name] = base.write(args.output / name, records)
        longest[name] = max(lengths)
    shutil.copy(args.source / "verifiers.json", args.output / "verifiers.json")

    manifest = {
        "source": str(args.source),
        "source_train_sha256": hashlib.sha256((args.source / "train.jsonl").read_bytes()).hexdigest(),
        "source_template_name": args.source_template,
        "template_name": args.template,
        "template": new("{problem}"),
        "required_miles_stop": base.TEMPLATES[args.template][1],
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": hashlib.sha256((Path(args.tokenizer) / "tokenizer.json").read_bytes()).hexdigest(),
        "max_prompt_tokens": args.max_prompt_tokens,
        "longest_prompt_tokens": longest,
        "artifacts": artifacts,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
