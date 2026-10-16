"""Verify and copy historical packing indices once, without reserving GPUs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from olmo_core import data as oc_data

from open_instruct import dataset_transformation, olmo_core_utils

SOURCES = {
    "original": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-hillclimb-1895/h025-aligned-train-full-s34521",
    "atomic": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/runs/h030-think-noemo-s34521",
}
CACHE_ROOT = Path("/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    reports = {}
    for kind, source in SOURCES.items():
        tokenizer = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="allenai/olmo-3-tokenizer-instruct-dev",
            chat_template_name="olmo123",
            reserved_slot_tokens=["<think>", "</think>"] if kind == "atomic" else None,
        )
        cache = CACHE_ROOT / ("3f12323c3b-6068a350" if kind == "atomic" else "062b8a3d20-6068a350")
        work = args.root / f"packing-{kind}"
        dataset = oc_data.NumpyPackedFSLDatasetConfig(
            tokenizer=olmo_core_utils.to_oc_tokenizer_config(tokenizer),
            work_dir=str(work),
            paths=[str(cache / "token_ids_part_*.npy")],
            expand_glob=True,
            label_mask_paths=[str(cache / "labels_mask_part_*.npy")],
            generate_doc_lengths=True,
            long_doc_strategy=oc_data.LongDocStrategy.truncate,
            sequence_length=65536,
        ).build()
        assert len(dataset.paths) == 85, f"Unexpected shard count: {len(dataset.paths)}"
        files = []
        # Core derives these names from source paths, sizes, sequence length,
        # truncation strategy and index dtype. Require every expected file.
        for path in dataset.paths:
            for target in (
                dataset._get_document_indices_path(path),
                dataset._get_instance_offsets_path(path),
                dataset._get_docs_by_instance_path(path),
            ):
                original = Path(source) / "dataset-common" / target.name
                if not original.is_file():
                    raise FileNotFoundError(f"Historical index unavailable: {original}; refusing to repack on GPUs")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest() != original_hash:
                    raise RuntimeError(f"Copy hash mismatch: {target}")
                files.append({"name": target.name, "sha256": original_hash, "bytes": target.stat().st_size})
        dataset.prepare()
        reports[kind] = {"source": source, "files": files, "instances": len(dataset)}
        print(f"Verified {kind}: {len(files)} copied indices, {len(dataset)} packed sequences", flush=True)
    for name in (
        "gate-reference",
        "gate-probed",
        "gate-resumed",
        "control",
        "rounding-only",
        "atomic-safe",
        "atomic-historical",
        "control-repeat",
    ):
        kind = "atomic" if name.startswith("atomic-") else "original"
        directory = args.root / name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "dataset-common"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing packing cache: {target}")
        target.symlink_to(args.root / f"packing-{kind}" / "dataset-common", target_is_directory=True)
    result = json.dumps(reports, indent=2) + "\n"
    (args.root / "packing-receipt.json").write_text(result)
    Path("/output").mkdir(exist_ok=True)
    Path("/output/packing-receipt.json").write_text(result)


if __name__ == "__main__":
    main()
