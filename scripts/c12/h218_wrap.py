"""Ledger H218/H219 (redsearcher continuation): olmo_core_finetune.py as a continuation of an SFT checkpoint.

  torchrun ... scripts/c12/h218_wrap.py <olmo_core_finetune.py arguments>

Three patches, each logged, then scripts/c12/h161_smoke_wrap.py runs (its hooks are off unless their variables are set):

1. H218_CACHE_ROOT: setup_tokenizer_and_cache forces --local_cache_dir to oe-adapt's deletable cache inside Beaker;
   the run's cache lives under this root instead (as H194's wrapper did for its root). The cache name is still the
   config hash, so EXPECTED_NUMPY_CACHE guards it as usual.
2. No promoted-row seeding. olmo_core_finetune.py seeds the <think> / </think> rows (100256 / 100266) from their
   source pieces on every non-resumed start, which is right for a base and wrong for a continuation: the start here
   is an SFT checkpoint whose rows were trained (H181), and the seed would overwrite them. The wrapper replaces
   initialize_promoted_token_embeddings with a no-op that only checks the promoted ids are the ones the start was
   trained with (H218_PROMOTED_IDS, default "100256,100266"), and drops the step check that assumes a seed.
3. H218_ASSERT_PROMOTED (default 1): after the tokenizer is set up, <think> and </think> must each encode to one
   token with those ids; otherwise the run dies before any data is read.
"""

import atexit
import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())
from open_instruct import olmo_core_utils  # noqa: E402

ROOT = os.environ["H218_CACHE_ROOT"]
PROMOTED = tuple(int(x) for x in os.environ.get("H218_PROMOTED_IDS", "100256,100266").split(","))
RANK = os.environ.get("RANK", "0")

_orig_setup = olmo_core_utils.setup_tokenizer_and_cache


def _setup(model_config, dataset_config, tc):
    tok = _orig_setup(model_config, dataset_config, tc)
    dataset_config.local_cache_dir = ROOT
    if os.environ.get("H218_ASSERT_PROMOTED", "1") == "1":
        ids = tuple(tok.encode(t, add_special_tokens=False) for t in ("<think>", "</think>"))
        if ids != tuple([i] for i in PROMOTED):
            raise RuntimeError(f"H218: promoted ids {ids} differ from the start's {PROMOTED}")
        print(f"H218-PROMOTED rank {RANK} <think>,</think> -> {PROMOTED} (as trained in the start)", flush=True)
    return tok


def _no_seed(train_module, tokenizer) -> int:
    print(f"H218-NOSEED rank {RANK}: promoted rows {PROMOTED} kept from the start checkpoint, not re-seeded", flush=True)
    return 0


olmo_core_utils.setup_tokenizer_and_cache = _setup
olmo_core_utils.initialize_promoted_token_embeddings = _no_seed
olmo_core_utils.PromotedRowStepCheck.before_seeding = classmethod(lambda cls, train_module, tokenizer: None)


@atexit.register
def _peak():
    if torch.cuda.is_available():
        g = 2**30
        print(f"H218-PEAK rank {RANK} max_alloc {torch.cuda.max_memory_allocated() / g:.1f} GiB "
              f"max_reserved {torch.cuda.max_memory_reserved() / g:.1f} GiB", flush=True)


sys.argv = ["/tmp/h161_smoke_wrap.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
