"""DPO plumbing test, JAX side: MaxText's pre-tokenized DPO path on a toy model vs open-instruct's formulas.

Builds open-instruct-style tokenized pairs (labels -100 outside the trained span, including untrained trailing
context), runs DPOPretokenizedFormatting -> MaxTextDPOTrainer._prepare_inputs -> Tunix dpo_loss_fn with a toy
"model" whose logits are tok_emb[token] + pos_emb[position], and saves everything plumbing_oi.py needs to recompute
the loss with open-instruct's own functions.
    python plumbing_jax.py out.npz
"""

import os
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# MaxText with our patch applied, and Tunix at the commit MaxText pins (a8d70582), both as source trees.
sys.path[:0] = [
    os.environ.get("TUNIX_SRC", "/opt/scratch/tunix-a8d"),
    os.environ.get("MAXTEXT_SRC", "/opt/scratch/mt-resume/src"),
]
from maxtext.input_pipeline.dpo_utils import DPOPretokenizedFormatting  # noqa: E402
from maxtext.trainers.post_train.dpo.train_dpo import MaxTextDPOTrainer  # noqa: E402
from tunix.sft.dpo.dpo_trainer import dpo_loss_fn  # noqa: E402

jax.config.update("jax_enable_x64", False)
V, L, PAD, BETA = 50, 24, 49, 5.0
rng = np.random.default_rng(0)


class Toy(nnx.Module):
    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.tok = nnx.Param(jnp.asarray(r.normal(size=(V, V)), jnp.float32))
        self.pos = nnx.Param(jnp.asarray(r.normal(size=(L, V)) * 0.3, jnp.float32))

    def __call__(self, input_tokens, positions, attention_mask, cache=None):
        assert attention_mask is None
        return self.tok[input_tokens] + self.pos[positions], None


def row(n_prompt, n_resp, n_tail):
    ids = rng.integers(0, PAD, size=n_prompt + n_resp + n_tail)
    labels = np.full_like(ids, -100)
    labels[n_prompt : n_prompt + n_resp] = ids[n_prompt : n_prompt + n_resp]
    return ids, labels


pairs = []
for n_prompt, (rc, rr), tail in [(5, (6, 9), 0), (3, (14, 2), 0), (8, (4, 4), 3), (2, (21, 1), 0)]:
    c_ids, c_lab = row(n_prompt, rc, tail)
    r_ids, r_lab = row(n_prompt, rr, 0)
    r_ids[:n_prompt] = c_ids[:n_prompt]
    pairs.append(dict(chosen_input_ids=c_ids, chosen_labels=c_lab, rejected_input_ids=r_ids, rejected_labels=r_lab))

cols = ("chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels")
fmt = DPOPretokenizedFormatting(PAD, L, cols)
elems = [fmt.map(dict(p)) for p in pairs]
batch = {k: jnp.asarray(np.stack([e[k] for e in elems])) for k in elems[0]}

policy, ref = Toy(1), Toy(2)
out = {}
for loss_type in ("dpo", "dpo_norm"):
    fake = types.SimpleNamespace(loss_type=loss_type, _ref_model_exists=True, ref_model=ref)
    ex = MaxTextDPOTrainer._prepare_inputs(fake, batch)
    lo = dpo_loss_fn(policy, ex, algorithm="dpo", beta=BETA)
    out[f"loss_{loss_type}"] = float(lo.primary_loss.unreduced_sum / lo.primary_loss.denominator)
    print(loss_type, out[f"loss_{loss_type}"])

np.savez(
    sys.argv[1],
    tok_policy=np.asarray(policy.tok.value),
    pos_policy=np.asarray(policy.pos.value),
    tok_ref=np.asarray(ref.tok.value),
    pos_ref=np.asarray(ref.pos.value),
    beta=BETA,
    **{f"{k}_{i}": p[k] for i, p in enumerate(pairs) for k in cols},
    n=len(pairs),
    **out,
)
