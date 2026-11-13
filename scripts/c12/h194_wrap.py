"""Ledger H194 (c12r-long): olmo_core_finetune.py reading the frozen C12 cache at --max_seq_length 131072.

  torchrun ... scripts/c12/h194_wrap.py <olmo_core_finetune.py arguments>

The cache's dir name hashes max_seq_length, and setup_tokenizer_and_cache forces --local_cache_dir to oe-adapt inside
Beaker. Two patches, as H194's M0 ran them, point the run at C12R_CACHE_ROOT/numpy_sft/bf82a8fab8-0a2e7a2c-rowaligned,
a symlink to the frozen cache bf82a8fab8-6068a350-rowaligned (same tokens; its 23 rows cut at 65,536 stay cut).
Nothing else changes. Then scripts/c12/h161_smoke_wrap.py runs, whose hooks are off unless their variables are set.
"""

import atexit
import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
from open_instruct import dataset_transformation, olmo_core_utils  # noqa: E402

ROOT = os.environ["C12R_CACHE_ROOT"]
_orig_setup = olmo_core_utils.setup_tokenizer_and_cache


def _setup(model_config, dataset_config, tc):
    tok = _orig_setup(model_config, dataset_config, tc)
    dataset_config.local_cache_dir = ROOT
    return tok


olmo_core_utils.setup_tokenizer_and_cache = _setup
dataset_transformation.compute_config_hash = lambda dcs, tc: "bf82a8fab8"


@atexit.register
def _peak():
    if torch.cuda.is_available():
        g = 2**30
        print(f"C12R-PEAK rank {os.environ.get('RANK')} max_alloc {torch.cuda.max_memory_allocated() / g:.1f} GiB "
              f"max_reserved {torch.cuda.max_memory_reserved() / g:.1f} GiB", flush=True)


sys.argv = ["/tmp/h161_smoke_wrap.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
