"""Materialize the exact H077 sample prefix on CPU, before reserving training GPUs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from olmo_core import data
from olmo_core.data import data_loader
from transformers import AutoTokenizer

from open_instruct import packed_sample_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("kind", choices=("original", "atomic"))
    parser.add_argument("--batches", type=int, default=528)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batches <= 528 or not 1 <= args.workers <= 16:
        parser.error("Bounded preparation requires 1–528 batches and 1–16 workers")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.root / "control/tokenizer", local_files_only=True)
    # Core packing uses the unchanged EOS/padding IDs for either representation.
    token_config = data.TokenizerConfig(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None if tokenizer.bos_token_id == tokenizer.eos_token_id else tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    arm = "control" if args.kind == "original" else "atomic-safe"
    cache_name = "062b8a3d20-6068a350" if args.kind == "original" else "3f12323c3b-6068a350"
    cache = Path("/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft") / cache_name
    work = args.root / arm
    dataset = data.NumpyPackedFSLDatasetConfig(
        tokenizer=token_config,
        work_dir=str(work),
        paths=[str(cache / "token_ids_part_*.npy")],
        expand_glob=True,
        label_mask_paths=[str(cache / "labels_mask_part_*.npy")],
        generate_doc_lengths=True,
        long_doc_strategy=data.LongDocStrategy.truncate,
        sequence_length=65536,
    ).build()
    for paths in dataset._source_path_groups:
        for path in (
            dataset._get_document_indices_path(*paths),
            dataset._get_instance_offsets_path(*paths),
            dataset._get_docs_by_instance_path(*paths),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
    if len(dataset) != (345843 if args.kind == "original" else 345754):
        raise ValueError("Unexpected packed dataset size")
    loader = data_loader.NumpyFSLDataLoader(
        dataset,
        global_batch_size=1048576,
        work_dir=str(work),
        seed=34521,
        dp_world_size=16,
        dp_rank=0,
        collator=data.DataCollator(pad_token_id=dataset.pad_token_id, vocab_size=dataset.vocab_size),
    )
    loader._epoch = 1
    indices = loader._build_global_indices()
    if (args.kind == "original" or loader._global_indices_file.exists()) and not np.array_equal(
        np.fromfile(loader._global_indices_file, dtype=np.uint32), indices
    ):
        raise ValueError("Generated order differs from saved training order")
    output = args.root / f"sample-cache-{args.kind}"
    manifest = packed_sample_cache.build_cache(
        dataset,
        indices[: args.batches * 16],
        output,
        workers=args.workers,
        provenance={
            "kind": args.kind,
            "epoch": 1,
            "data_loader_seed": 34521,
            "batches": args.batches,
            "global_batch_tokens": 1048576,
            "global_order_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        },
    )
    result = Path("/output")
    result.mkdir(parents=True, exist_ok=True)
    (result / f"sample-cache-{args.kind}.json").write_text(
        json.dumps(
            {
                "path": str(output),
                "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
                "provenance": manifest["provenance"],
                "files": manifest["files"],
                "samples": len(manifest["rows"]),
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
