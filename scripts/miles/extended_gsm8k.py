"""Freeze the 500-update extension against the original immutable GSM8K selection."""

import argparse
import json
from pathlib import Path

from scripts.miles.prepare_gsm8k_parity import verify_preparation, write_immutable

SOURCE = Path("/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260910-core-megatron-v1")
ROOT = Path("/weka/oe-training-default/robertb/open-instruct/gsm8k-parity/20260911-abhishek-500-v1")
CAMPAIGN = "gsm8k-core-megatron-20260911-500-v1"
UPDATES = 500
EVAL_INTERVAL = 20
SAVE_INTERVAL = 100


def prepare(source=SOURCE, root=ROOT):
    source, root = Path(source), Path(root)
    if source.resolve() == root.resolve():
        raise ValueError("Extension must use a distinct campaign root")
    original = verify_preparation(source)
    root.mkdir(parents=True, exist_ok=True)
    # Keep the exact descriptor and native manifest, including their relative
    # paths, together. Read-only use of the old data never touches old outputs.
    for name in ("hf", "baseline", "tasks.toml", "train.jsonl", "eval.jsonl", "verifiers.json"):
        destination = root / name
        target = source / name
        if destination.is_symlink():
            if destination.resolve() != target.resolve():
                raise ValueError(f"Extension artifact points to another source: {name}")
        elif destination.exists():
            raise ValueError(f"Extension artifact is not the canonical symlink: {name}")
        else:
            destination.symlink_to(target, target_is_directory=target.is_dir())
    write_immutable(root / "preparation.json", (source / "preparation.json").read_bytes())
    verified = verify_preparation(root)
    if verified != original:
        raise ValueError("Extension preparation differs from original")
    protocol = dict(
        campaign=CAMPAIGN,
        updates=UPDATES,
        eval_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        source_root=str(source),
        train_prompts=400,
        prompt_groups_per_update=4,
        passes=5,
        shuffle=False,
        held_out_prompts=128,
        fresh_start=True,
        core_scoring_fix="290d2ca4521373bef0bf7fe4244673cc79dcc004",
        interpretation="Repeated exposure to the original400 prompts; not2000 unique prompts. "
        "Both arms start from SFT weights; Core runtime differs from the original100-update run.",
    )
    write_immutable(root / "extension.json", (json.dumps(protocol, indent=2) + "\n").encode())
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.root), indent=2))


if __name__ == "__main__":
    main()
