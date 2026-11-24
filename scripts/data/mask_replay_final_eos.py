"""Write a copy of a numpy SFT dir with no loss on each replay document's final <|endoftext|> (ledger H270).

    uv run python scripts/data/mask_replay_final_eos.py SRC_DIR OUT_DIR

A long-context replay dir (pretraining-style documents, each ending in <|endoftext|>, passed to
olmo_core_finetune.py with --extra_numpy_dirs) trains that final token as a target, which teaches the
model to stop after a long context. OUT_DIR gets new label masks, False on the final <|endoftext|> of
every document that does not start with <|im_start|>, and symlinks to everything else in SRC_DIR.
Train with --extra_numpy_dirs OUT_DIR instead of SRC_DIR: tokens, document boundaries, packing and
data order are unchanged, and runs that keep SRC_DIR are unaffected. Check the result with
scripts/data/check_replay_eos_mask.py.
"""

import argparse
import json

from open_instruct import numpy_dataset_conversion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_dir", help="Numpy dir to read (token_ids_part_*.npy, labels_mask_part_*.npy, .csv.gz).")
    parser.add_argument("out_dir", help="Dir to create; must not exist.")
    parser.add_argument("--eos_token_id", type=int, default=numpy_dataset_conversion.OLMO35_EOS_TOKEN_ID)
    parser.add_argument("--im_start_token_id", type=int, default=numpy_dataset_conversion.OLMO35_IM_START_TOKEN_ID)
    args = parser.parse_args()
    receipt = numpy_dataset_conversion.write_replay_eos_masked_dir(
        args.src_dir, args.out_dir, args.eos_token_id, args.im_start_token_id
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "parts"}, indent=1))


if __name__ == "__main__":
    main()
