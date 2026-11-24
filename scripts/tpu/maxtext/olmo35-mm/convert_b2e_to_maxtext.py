"""olmo_core_to_maxtext conversion of an OLMo-core multimodal DCP with <think>/</think> row seeding.

The script that converted B2e step 22240 into the reference run's start checkpoint
(gs://ai2-olmo-dev/post-training/checkpoints/olmo35-mm-b2e-emocausal-step22240-thinkseed, 2026-11-20; its
seed_receipt.json records the argv), copied from the session that ran it. The converter it wraps is
src/maxtext/checkpoint_conversion/standalone_scripts/convert_olmo35_multimodal_ckpt.py, byte-identical at
jason718/maxtext@2b8446d3ef0605cffdabd3a78d7da422f2e288db, @217c84582d2f985c9b2e9d3a7067e8423035c19e and
@e3543ef27552d042a54c74ea89cf4d6b06ca4520; the seeding rows come from allenai/OLMo-core@fd9bba1ba
(olmo_core.train.callbacks.promoted_rows.OLMO35_THINK_ROWS). Run through convert_b2e.sh.

    python convert_b2e_to_maxtext.py [--receipt seed_receipt.json] <converter args>

Runs MaxText's convert_olmo35_multimodal_checkpoint (--mode olmo_core_to_maxtext) unchanged, except that right after the
fp32 LM weights are loaded from the DCP, the promoted-token rows are seeded exactly as OLMo-core's
PromotedTokenRowsCallback / MultimodalTrainModule.seed_token_rows_from_pieces does before SFT step 1:
  for each matrix in (lm.embeddings.weight, lm.lm_head.w_out.weight)  [untied]
    means = {t: W[srcs].float().mean(0) for t, srcs in ROWS}   # all sources read before any write
    W[t] = means[t]
ROWS = OLMO35_THINK_ROWS = [(100256, [14023, 771, 29]), (100266, [524, 27963, 29])].
Writes a seed receipt JSON (sha256 of the fp32 rows before/after) to --receipt.
"""

import hashlib
import json
import sys

import torch
from maxtext.checkpoint_conversion.standalone_scripts import convert_olmo35_multimodal_ckpt as conv
from olmo_core.train.callbacks.promoted_rows import OLMO35_THINK_ROWS

MATRICES = ("embeddings.weight", "lm_head.w_out.weight")
RECEIPT = {"rows": [[t, list(s)] for t, s in OLMO35_THINK_ROWS], "matrices": {}}


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().float().numpy().tobytes()).hexdigest()


_orig_load = conv.load_olmo_core_checkpoint_weights


def _load_and_seed(checkpoint_dir, experiment_config):
    lm_state, vis_state = _orig_load(checkpoint_dir, experiment_config)
    with torch.no_grad():
        for name in MATRICES:
            w = lm_state[name]
            assert w.dtype == torch.float32 and w.shape[0] == 100352, (name, w.dtype, w.shape)
            info = {"shape": list(w.shape), "rows": {}}
            means = {int(t): w[[int(i) for i in s]].to(torch.float32).mean(dim=0) for t, s in OLMO35_THINK_ROWS}
            for t, mean in means.items():
                before = w[t].clone()
                w[t] = mean.to(w.dtype)
                info["rows"][str(t)] = {
                    "pieces": [int(i) for i in dict(OLMO35_THINK_ROWS)[t]],
                    "fp32_sha256_before": _sha(before),
                    "fp32_sha256_after": _sha(w[t]),
                    "l2_before": float(before.norm()),
                    "l2_after": float(w[t].norm()),
                    "l2_change": float((w[t] - before).norm()),
                }
            RECEIPT["matrices"][name] = info
    print("SEEDED", json.dumps(RECEIPT, indent=1), flush=True)
    return lm_state, vis_state


conv.load_olmo_core_checkpoint_weights = _load_and_seed

if __name__ == "__main__":
    argv = sys.argv[1:]
    receipt_path = None
    if "--receipt" in argv:
        i = argv.index("--receipt")
        receipt_path = argv[i + 1]
        del argv[i : i + 2]
    RECEIPT["converter_argv"] = argv
    conv.main(argv)
    if receipt_path:
        with open(receipt_path, "w") as f:
            json.dump(RECEIPT, f, indent=1)
