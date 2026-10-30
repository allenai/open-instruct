"""Run open_instruct/olmo_core_finetune.py with smoke-only hooks (ledger H161 GPU companion smoke).

  torchrun ... scripts/c12/h161_smoke_wrap.py <olmo_core_finetune.py arguments>

Run from the open-instruct root. Every hook is off unless its variable is set, so with none set this is exactly
olmo_core_finetune.py.

  H161_DUMP_DIR=<dir>     each rank saves its first batch of epoch 1 (every tensor in the batch: input_ids, label_mask,
                          doc_lens, ...) to <dir>/rankNN.npz, and logs the packed-instance count and batches per epoch.
                          scripts/c12/h161_refce.py forwards those batches through the HF base (step-1 CE gate).
  H161_NO_FINAL_SAVE=1    skip the checkpointer's end-of-training save, a full DCP of ~200 GB that a 30-step smoke
                          does not need.
  H161_BATCH_HASH=1       each rank logs one line per batch: H161-BH rank R epoch E batch B sha <sha256 over every
                          tensor in the batch, keys sorted> index <global instance indices>. The resume gate (G3)
                          compares these lines across the interrupted and resumed attempts.
  H161_HOLD_AT_STEP=N     resume gate provocation. On an attempt that starts with no <output_dir>/h161_hold_N.done,
                          every rank stops after the checkpointer's post_train_batch of step N (rank 0 writes the
                          marker first and logs H161-HOLD), and waits H161_HOLD_MINUTES (default 20) to be preempted
                          (beaker job preempt <job>). Not preempted by then: exit 3, which Beaker's retry, if set,
                          relaunches into the same resume path. A later attempt finds the marker and does not hold.
"""

import hashlib
import os
import runpy
import sys
import time

import numpy as np
import torch
from olmo_core.data import data_loader as oc_data_loader
from olmo_core.train.callbacks import checkpointer as oc_checkpointer

ROOT = os.getcwd()
sys.path.insert(0, ROOT)

DUMP_DIR = os.environ.get("H161_DUMP_DIR")
BATCH_HASH = os.environ.get("H161_BATCH_HASH") == "1"
RANK = int(os.environ.get("RANK", "0"))


def _batch_hash(batch) -> str:
    h = hashlib.sha256()
    for k in sorted(batch):
        v = batch[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(f"{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


if DUMP_DIR or BATCH_HASH:
    _orig_iter = oc_data_loader.TextDataLoaderBase.__iter__

    def _iter_and_dump(self):
        for batch in _orig_iter(self):
            if BATCH_HASH:
                idx = batch.get("index")
                idx = idx.tolist() if torch.is_tensor(idx) else idx
                print(
                    f"H161-BH rank {RANK} epoch {self.epoch} batch {self.batches_processed} "
                    f"sha {_batch_hash(batch)} index {idx}",
                    flush=True,
                )
            if DUMP_DIR and self.epoch == 1 and self.batches_processed == 1:
                rank = RANK
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

HOLD_AT = int(os.environ.get("H161_HOLD_AT_STEP", "0"))
if HOLD_AT:
    _out = sys.argv[sys.argv.index("--output_dir") + 1]
    _marker = os.path.join(_out, f"h161_hold_{HOLD_AT}.done")
    # Decided once, before training, so every rank of an attempt agrees.
    _armed = not os.path.exists(_marker)
    _steps = sorted(d for d in (os.listdir(_out) if os.path.isdir(_out) else []) if d.startswith("step"))
    print(f"H161-ATTEMPT rank {RANK} hold_at {HOLD_AT} armed {_armed} step_dirs {_steps}", flush=True)
    _orig_ptb = oc_checkpointer.CheckpointerCallback.post_train_batch

    def _ptb_then_hold(self):
        _orig_ptb(self)
        if _armed and self.step == HOLD_AT:
            if RANK == 0:
                with open(_marker, "w") as f:
                    f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
            print(
                f"H161-HOLD rank {RANK} step {self.step} latest_ckpt {self._latest_checkpoint_path} "
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
                flush=True,
            )
            deadline = time.time() + 60 * float(os.environ.get("H161_HOLD_MINUTES", "20"))
            while time.time() < deadline:
                time.sleep(10)
            print(f"H161-HOLD-EXPIRED rank {RANK}", flush=True)
            os._exit(3)

    oc_checkpointer.CheckpointerCallback.post_train_batch = _ptb_then_hold

sys.argv = [os.path.join(ROOT, "open_instruct", "olmo_core_finetune.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
