"""Replay a 4x4x8 data host using the pinned image, following H083's CPU replay."""

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import datasets
import jax
import numpy as np
from absl import flags
from maxtext.input_pipeline import hf_data_processing

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=int, required=True, choices=range(8))
parser.add_argument("--arm", required=True, choices=("control", "uniform"))
parser.add_argument("--data-root", type=Path, default=Path("/corpus"))
parser.add_argument("--output", type=Path, default=Path("/work/replay"))
args = parser.parse_args()
host = args.host
out = args.output / args.arm
out.mkdir(parents=True, exist_ok=True)
assert not (out / f"h{host}.json").exists(), "Do not overwrite a replay receipt"
flags.FLAGS(sys.argv[:1])
jax.process_count = lambda: 32
hf_data_processing.local_rows_trained = lambda *args, **kwargs: [True, True, False, False]
hf_data_processing.multihost_dataloading.MultiHostDataLoadIterator = lambda loader, *args, **kwargs: loader
config = types.SimpleNamespace(elastic_enabled=False, training_objective="causal_lm", micro_batch_size_to_train_on=16)
dataset = datasets.load_dataset(
    "parquet", data_files=str(args.data_root / args.arm / "*.parquet"), split="train", streaming=True
)
loader = hf_data_processing.preprocessing_pipeline(
    dataloading_host_index=host,
    dataloading_host_count=8,
    global_mesh=types.SimpleNamespace(size=128),
    dataset=dataset,
    config=config,
    data_column_names=["messages"],
    tokenize=True,
    tokenizer_path=str(args.data_root / "tokenizer"),
    hf_access_token=None,
    global_batch_size=128,
    max_target_length=65536,
    shuffle=True,
    data_shuffle_seed=0,
    add_bos=True,
    add_eos=True,
    packing=True,
    generate_padding_batch=False,
    use_dpo=False,
    use_sft=True,
    use_tunix_gradient_accumulation=True,
    num_microbatches=1,
    sft_train_on_completion_only=True,
    chat_template_path="",
    max_segments_per_seq=-1,
    num_epoch=2,
    grain_worker_count=0,
)
receipt = {"host": host, "arm": args.arm, "data_root": str(args.data_root), "batches": []}
for step, batch in enumerate(loader, 1):
    digest = hashlib.sha256()
    for key in sorted(batch):
        array = np.ascontiguousarray(np.asarray(batch[key]))
        digest.update(f"{key}:{array.dtype.str}:{array.shape}".encode())
        digest.update(array.tobytes())
    receipt["batches"].append(
        {
            "step": step,
            "sha256": digest.hexdigest(),
            "loss_tokens": int((np.asarray(batch["targets_segmentation"])[:2] > 0).sum()),
        }
    )
    if step == 1:
        np.savez(out / f"h{host}.step1.npz", **{key: np.asarray(value)[:2] for key, value in batch.items()})
    if step == 20:
        break
assert len(receipt["batches"]) == 20, "Input stream ended before the registered 20-batch window"
(out / f"h{host}.json").write_text(json.dumps(receipt, indent=2))
print("REPLAY-COMPLETE", host, len(receipt["batches"]), flush=True)
