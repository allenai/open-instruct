"""Freeze H077 probes from a pinned MATH-500 snapshot before any GPU results."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from open_instruct import dataset_transformation

DATA_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
TOKENIZER_REVISION = "55f211dfda3974963b869e490617447045069a64"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    random.Random(7601).shuffle(rows)
    original = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path="allenai/olmo-3-tokenizer-instruct-dev",
        tokenizer_revision=TOKENIZER_REVISION,
        chat_template_name="olmo123",
    ).tokenizer
    atomic = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path="allenai/olmo-3-tokenizer-instruct-dev",
        tokenizer_revision=TOKENIZER_REVISION,
        chat_template_name="olmo123",
        reserved_slot_tokens=["<think>", "</think>"],
    ).tokenizer
    documents = []
    for index, row in enumerate(rows[:192]):
        kind = "ordinary" if index < 128 else "boundary"
        text = row["problem"]
        if kind == "boundary":
            # Both tags remain present even for long reference solutions.
            text = text[:1600] + "\n<think>\n" + row["solution"][:3600] + "\n</think>\n" + str(row["answer"])[:500]
        encodings = {}
        for name, tokenizer in (("original", original), ("atomic", atomic)):
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if kind == "ordinary":
                ids, offsets = ids[:2048], offsets[:2048]
            else:
                assert len(ids) <= 2048, (index, len(ids))
            encodings[name + "_ids"] = ids
            encodings[name + "_offsets"] = offsets
        if kind == "ordinary":
            assert "<think>" not in text and "</think>" not in text
            assert encodings["original_ids"] == encodings["atomic_ids"]
        documents.append({"id": row["unique_id"], "kind": kind, "text": text, **encodings})
    assert len({document["id"] for document in documents}) == 192
    manifest = {
        "source": "HuggingFaceH4/MATH-500",
        "revision": DATA_REVISION,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection": "Python random.Random(7601).shuffle; first 128 problems ordinary, next 64 boundary",
        "tokenizer": "allenai/olmo-3-tokenizer-instruct-dev",
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files_hash": dataset_transformation.get_files_hash_if_exists(
            "allenai/olmo-3-tokenizer-instruct-dev",
            TOKENIZER_REVISION,
            filenames=["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json"],
        ),
        "documents": documents,
    }
    args.output.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
