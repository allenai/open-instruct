"""Check MaxTextDPOTrainer's attention-mask-free path on a real (tiny, random) Olmo 3 MaxText model on CPU.

Per-token log-probs of the trained tokens from the right-padded DPO batch (attention from `!= pad_id`) must equal
those from unpadded single-sequence forwards.
"""

import os
import sys
import types

# MaxText with our patch applied, and Tunix at the commit MaxText pins (a8d70582), both as source trees.
sys.path[:0] = [
    os.environ.get("TUNIX_SRC", "/opt/scratch/tunix-a8d"),
    os.environ.get("MAXTEXT_SRC", "/opt/scratch/mt-resume/src"),
]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax.core.spmd import logical_axis_rules  # noqa: E402
from maxtext.configs import pyconfig  # noqa: E402
from maxtext.input_pipeline.dpo_utils import DPOPretokenizedFormatting  # noqa: E402
from maxtext.trainers.post_train.dpo.train_dpo import MaxTextDPOTrainer  # noqa: E402
from maxtext.utils import model_creation_utils  # noqa: E402
from tunix.rl import common  # noqa: E402

L, PAD = 64, 100277
cfg = pyconfig.initialize(
    [
        "",
        os.path.join(
            os.environ.get("MAXTEXT_SRC", "/opt/scratch/mt-resume/src"), "maxtext/configs/post_train/dpo.yml"
        ),
        "model_name=olmo3-7b",
        "override_model_config=true",
        "base_num_decoder_layers=4",
        "base_emb_dim=128",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=32",
        "base_mlp_dim=256",
        f"max_target_length={L}",
        "per_device_batch_size=4",
        "attention=dot_product",
        "skip_jax_distributed_system=true",
        "dpo.pretokenized=true",
        "tokenize_train_data=false",
        "dpo.loss_type=dpo_norm",
        "enable_checkpointing=false",
        "tokenizer_path=/root/tpu-posttrain/tok-olmo3-instruct-dev-fixed",
        "train_data_columns=['chosen_input_ids','chosen_labels','rejected_input_ids','rejected_labels']",
        "dtype=float32",
        "weight_dtype=float32",
        "run_name=padcheck",
        "base_output_directory=/tmp/padcheck",
    ]
)
model, mesh = model_creation_utils.from_pretrained(cfg, wrap_with_tunix_adapter=True, tokenizer_pad_id=PAD)

rng = np.random.default_rng(0)
pairs = []
for n_prompt, rc, rr, tail in [(5, 20, 9, 0), (12, 3, 40, 4), (2, 61, 1, 0)]:
    c = rng.integers(0, 100000, size=n_prompt + rc + tail)
    r = rng.integers(0, 100000, size=n_prompt + rr)
    r[:n_prompt] = c[:n_prompt]
    cl = np.full_like(c, -100)
    cl[n_prompt : n_prompt + rc] = c[n_prompt : n_prompt + rc]
    rl = np.full_like(r, -100)
    rl[n_prompt:] = r[n_prompt:]
    pairs.append(dict(chosen_input_ids=c, chosen_labels=cl, rejected_input_ids=r, rejected_labels=rl))
cols = ("chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels")
elems = [DPOPretokenizedFormatting(PAD, L, cols).map(dict(p)) for p in pairs]
batch = {k: jnp.asarray(np.stack([e[k] for e in elems])) for k in elems[0]}

with jax.set_mesh(mesh), logical_axis_rules(cfg.logical_axis_rules):
    fake = types.SimpleNamespace(loss_type="dpo", _ref_model_exists=False, ref_model=None)
    ex = MaxTextDPOTrainer._prepare_inputs(fake, batch)
    padded = common.get_per_token_logps(model, ex.input_ids, ex.positions, None, ex.logits_to_keep)
    worst = 0.0
    seqs = [p["chosen_input_ids"] for p in pairs] + [p["rejected_input_ids"] for p in pairs]
    for i, s in enumerate(seqs):
        x = jnp.asarray(s[None].astype(np.int32))
        pos = jnp.arange(len(s))[None]
        single = common.get_per_token_logps(model, x, pos, None, len(s) - 1)[0]
        diff = float(jnp.max(jnp.abs(single - padded[i, : len(s) - 1])))
        worst = max(worst, diff)
        print(f"seq {i} len {len(s)}: max |dlogp| {diff:.2e}")
print(f"WORST {worst:.2e}", "OK" if worst < 1e-4 else "MISMATCH")
