"""Run open_instruct/olmo_core_finetune.py with two smoke-only hooks (ledger H161 GPU companion smoke).

  torchrun ... scripts/c12/h161_smoke_wrap.py <olmo_core_finetune.py arguments>

Run from the open-instruct root. Both hooks are off unless their variable is set, so with neither set this is exactly
olmo_core_finetune.py.

  H161_DUMP_DIR=<dir>     each rank saves its first batch of epoch 1 (every tensor in the batch: input_ids, label_mask,
                          doc_lens, ...) to <dir>/rankNN.npz, and logs the packed-instance count and batches per epoch.
                          scripts/c12/h161_refce.py forwards those batches through the HF base (step-1 CE gate).
  H161_NO_FINAL_SAVE=1    skip the checkpointer's end-of-training save, a full DCP of ~200 GB that a 30-step smoke
                          does not need.
"""

import os
import runpy
import sys

import numpy as np
import torch
from olmo_core.data import data_loader as oc_data_loader
from olmo_core.train.callbacks import checkpointer as oc_checkpointer

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

DUMP_DIR = os.environ.get("H161_DUMP_DIR")
if DUMP_DIR:
    _orig_iter = oc_data_loader.TextDataLoaderBase.__iter__

    def _iter_and_dump(self):
        for batch in _orig_iter(self):
            if self.epoch == 1 and self.batches_processed == 1:
                rank = int(os.environ.get("RANK", "0"))
                os.makedirs(DUMP_DIR, exist_ok=True)
                arrs = {k: v.detach().cpu().numpy() for k, v in batch.items() if torch.is_tensor(v)}
                np.savez(os.path.join(DUMP_DIR, f"rank{rank:02d}.npz"), **arrs)
                print(
                    f"H161-DUMP rank {rank} epoch {self.epoch} instances {len(self.dataset)} "
                    f"total_batches {self.total_batches} "
                    f"keys {[(k, arrs[k].shape, str(arrs[k].dtype)) for k in sorted(arrs)]}",
                    flush=True,
                )
            yield batch

    oc_data_loader.TextDataLoaderBase.__iter__ = _iter_and_dump

if os.environ.get("H161_NO_FINAL_SAVE") == "1":
    oc_checkpointer.CheckpointerCallback.post_train = lambda self: None

sys.argv = [os.path.join(ROOT, "open_instruct", "olmo_core_finetune.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
