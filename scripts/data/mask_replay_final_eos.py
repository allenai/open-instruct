"""Write a copy of a replay dir with no loss on each replay document's final <|endoftext|> (ledger H270).

    uv run python scripts/data/mask_replay_final_eos.py SRC_DIR OUT_DIR

A long-context replay dir (pretraining-style documents, each ending in <|endoftext|>, passed to
olmo_core_finetune.py with --extra_numpy_dirs) trains that final token as a target, which teaches the
model to stop after a long context. SRC_DIR must be written by the ledger's replay build
(experiments/c11/c11_replay.py build): its manifest.jsonl gives the document boundaries, which must
equal the .csv.gz rows and keep every document inside one part. OUT_DIR gets new label masks, False
on the final <|endoftext|> of every document that does not start with <|im_start|>, a new
dataset_statistics.json (loss tokens from the new masks, row-aligned marker) and symlinks to SRC_DIR's
token parts, .csv.gz parts and manifest. Train with --extra_numpy_dirs OUT_DIR instead of SRC_DIR:
tokens, boundaries, packing and data order are unchanged. Check it with
scripts/data/check_replay_eos_mask.py.
"""

import argparse
import json

from olmo_core import data as oc_data

from open_instruct import numpy_dataset_conversion, olmo_core_finetune


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_dir", help="Numpy dir to read (token_ids_part_*.npy, labels_mask_part_*.npy, .csv.gz).")
    parser.add_argument("out_dir", help="Dir to create; must not exist.")
    parser.add_argument(
        "--expected_manifest_sha256",
        default=numpy_dataset_conversion.H182_REPLAY_MANIFEST_SHA256,
        help="sha256 SRC_DIR's manifest.jsonl must have. Default: H182's replay dir's.",
    )
    parser.add_argument("--max_seq_length", type=int, default=65536, help="For the startup check's truncation counts.")
    parser.add_argument("--eos_token_id", type=int, default=numpy_dataset_conversion.OLMO35_EOS_TOKEN_ID)
    parser.add_argument("--im_start_token_id", type=int, default=numpy_dataset_conversion.OLMO35_IM_START_TOKEN_ID)
    args = parser.parse_args()
    stats = numpy_dataset_conversion.write_replay_eos_masked_dir(
        args.src_dir, args.out_dir, args.expected_manifest_sha256, args.eos_token_id, args.im_start_token_id
    )
    # The training job's own startup validation of an --extra_numpy_dirs entry, with the Olmo tokenizer's
    # OLMo-core config (olmo_core_utils.to_oc_tokenizer_config gives dolma2() for it) and metadata boundaries.
    tokenizer = oc_data.TokenizerConfig.dolma2()
    token_dtype = oc_data.NumpyPackedFSLDatasetConfig(tokenizer=tokenizer, paths=[], sequence_length=1).get_dtype()
    checked = olmo_core_finetune._check_extra_numpy_dir(
        args.out_dir, tokenizer, token_dtype, args.max_seq_length, require_metadata=True
    )
    # Main runs this only for multi-part dirs; OUT always carries the statistics, so check them either way.
    olmo_core_finetune._check_row_aligned_cache(args.out_dir)
    if checked.num_loss_tokens != stats["overall_statistics"]["trainable_tokens"]:
        raise ValueError(f"startup check counts {checked.num_loss_tokens:,} loss tokens; statistics say otherwise.")
    change = stats["configuration"]["replay_eos_mask"]
    print(json.dumps({**{k: v for k, v in change.items() if k != "parts"}, **stats["overall_statistics"]}, indent=1))
    print(f"Startup check PASS: {checked}")


if __name__ == "__main__":
    main()
