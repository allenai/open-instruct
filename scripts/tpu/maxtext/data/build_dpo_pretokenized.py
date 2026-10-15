"""Write open-instruct's tokenized preference dataset as parquet shards for MaxText's dpo.pretokenized path.

Takes the same flags as open_instruct/dpo.py and builds the dataset exactly as it does (get_cached_dataset_tulu with
the DPO transforms, then dataset.shuffle(seed)), so tokens and loss masks match open-instruct by construction. The
shards keep that global order; MaxText's streaming shuffle only permutes shards plus a buffer.
  python build_dpo_pretokenized.py --out_dir <dir> --n_shards 128 <dpo.py flags: --mixer_list, --max_seq_length, ...>
"""

import argparse
import os

from open_instruct import dataset_transformation, dpo_utils, olmo_core_utils
from open_instruct.utils import ArgumentParserPlus

COLUMNS = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels", "index"]


def main() -> None:
    own = argparse.ArgumentParser()
    own.add_argument("--out_dir", required=True)
    own.add_argument("--n_shards", type=int, required=True)
    ours, flags = own.parse_known_args()
    parser = ArgumentParserPlus((dpo_utils.DPOExperimentConfig, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses(flags)
    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    dataset = olmo_core_utils.load_dataset_distributed(args, tc, transform_fn_args, is_main_process=True)
    dataset = dataset.shuffle(seed=args.seed).select_columns(COLUMNS).flatten_indices()
    os.makedirs(ours.out_dir, exist_ok=True)
    for i in range(ours.n_shards):
        shard = dataset.shard(ours.n_shards, i, contiguous=True)
        shard.to_parquet(f"{ours.out_dir}/train-{i:05d}-of-{ours.n_shards:05d}.parquet")
    print(dataset.num_rows, "pairs ->", ours.n_shards, "shards in", ours.out_dir)


if __name__ == "__main__":
    main()
