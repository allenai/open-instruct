"""Read-only step-2 forwards with EP8 post-step-1 input chat rows."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp
from jax import sharding

INPUT_KEY = "['token_embedder']['embedding'].value"
TOKEN_IDS = (100264, 100265)
MODES = ("unchanged", "im_start", "im_end", "both")


def load_donor(path):
    value = json.loads(path.read_text())
    assert value["token_ids"] == list(TOKEN_IDS)
    rows = np.asarray(value["rows"], dtype=np.float32)
    assert rows.shape == (2, 1024) and np.isfinite(rows).all()
    assert hashlib.sha256(rows.tobytes()).hexdigest() == value["rows_sha256"]
    return rows


def replace_rows(params, donor, mode):
    found = []

    def replace(path, value):
        if jax.tree_util.keystr(path) != INPUT_KEY:
            return value
        found.append(True)
        ids = jnp.asarray(TOKEN_IDS)
        selected = (jnp.asarray(mode) & jnp.asarray([1, 2])) != 0
        replacement = jnp.where(selected[:, None], donor.astype(value.dtype), value[ids])
        return value.at[ids].set(replacement)

    result = jax.tree_util.tree_map_with_path(replace, params)
    assert len(found) == 1, "Expected exactly one input embedding table"
    return result


def measure(loss_wrapper, params, rest, inputs, optimizer_step, donor):
    def run(_):
        def body(carry, mode):
            changed = replace_rows(params, donor, mode)
            _, (aux, _) = loss_wrapper(changed, rest, **inputs)
            parts = jnp.stack([aux["xent_sum"], aux["total_weights"], aux["z_loss"]])
            parts = jax.lax.with_sharding_constraint(parts, sharding.PartitionSpec())
            return carry, parts

        return jax.lax.scan(body, None, jnp.arange(4))[1]

    return jax.lax.cond(optimizer_step == 1, run, lambda _: jnp.zeros((4, 3), dtype=jnp.float32), None)


def emit(step, values):
    if step != 2:
        return
    local = values.addressable_data(0) if isinstance(values, jax.Array) else values
    array = np.asarray(local)
    assert array.shape == (4, 3) and np.isfinite(array).all()
    rows = []
    for mode, (xent, tokens, z) in zip(MODES, array, strict=True):
        assert tokens > 0
        rows.append({"mode": mode, "ce": float(xent / (tokens + 1e-8) - z), "tokens": int(tokens), "z": float(z)})
    print("H127-CHAT-ROW-CF " + json.dumps({"step": step, "rows": rows}), flush=True)


DONOR_PATH = Path(__file__).with_name("chat-row-donor.json")
